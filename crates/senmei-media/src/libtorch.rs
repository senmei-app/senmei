use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum LibTorchBackend {
    Cpu,
    Cuda,
    Rocm,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct LibTorchInfo {
    pub backend: LibTorchBackend,
    pub downloaded: bool,
    pub version: Option<String>,
    pub driver: Option<String>,
    pub devices: Vec<String>,
    pub path: Option<String>,
}

pub fn detect_backend() -> LibTorchBackend {
    let nvidia = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if nvidia {
        return LibTorchBackend::Cuda;
    }
    if Path::new("/dev/kfd").exists() {
        return LibTorchBackend::Rocm;
    }
    LibTorchBackend::Cpu
}

fn cmd_out(cmd: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
}

fn libtorch_version(dir: &Path) -> Option<String> {
    let path = dir.join("build-version");
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn driver_version(backend: LibTorchBackend) -> Option<String> {
    match backend {
        LibTorchBackend::Cuda => cmd_out(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        )
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .filter(|s| !s.is_empty()),
        LibTorchBackend::Rocm => cmd_out("rocm-smi", &["--showdriverversion"]).and_then(|s| {
            s.lines()
                .find(|l| l.to_ascii_lowercase().contains("version"))
                .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()))
        }),
        LibTorchBackend::Cpu => None,
    }
}

fn devices(backend: LibTorchBackend) -> Vec<String> {
    match backend {
        LibTorchBackend::Cuda => cmd_out("nvidia-smi", &["-L"])
            .map(|s| {
                s.lines()
                    .filter_map(|l| {
                        let name = l.split(": ").nth(1)?;
                        Some(
                            name.split(" (UUID")
                                .next()
                                .unwrap_or(name)
                                .trim()
                                .to_string(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
        LibTorchBackend::Rocm => cmd_out("rocm-smi", &["--showproductname"])
            .map(|s| {
                s.lines()
                    .filter_map(|l| l.split_once("Product Name:").map(|(_, n)| n.trim().to_string()))
                    .filter(|n| !n.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        LibTorchBackend::Cpu => Vec::new(),
    }
}

pub fn libtorch_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("libtorch")
}

pub fn status(data_dir: &Path) -> LibTorchInfo {
    let dir = libtorch_dir(data_dir);
    let downloaded = dir.join("lib").join("libtorch.so").exists()
        || dir.join("lib").join("libtorch.dll").exists();
    LibTorchInfo {
        backend: detect_backend(),
        downloaded,
        version: downloaded.then(|| libtorch_version(&dir)).flatten(),
        driver: driver_version(detect_backend()),
        devices: devices(detect_backend()),
        path: downloaded.then(|| dir.to_string_lossy().into_owned()),
    }
}

fn url(backend: LibTorchBackend) -> Option<&'static str> {
    match backend {
        LibTorchBackend::Cpu => Some(
            "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip",
        ),
        LibTorchBackend::Cuda => Some(
            "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip",
        ),
        LibTorchBackend::Rocm => Some(
            "https://download.pytorch.org/libtorch/rocm5.6/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Brocm5.6.zip",
        ),
    }
}

fn fetch(url: &str, dest: &Path, on_progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    let resp = ureq::get(url).call().map_err(|e| Error::command_failed(format!("download failed: {e}")))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(dest).map_err(|e| Error::command_failed(e.to_string()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::command_failed(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| Error::command_failed(e.to_string()))?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }
    Ok(())
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive).map_err(|e| Error::command_failed(e.to_string()))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| Error::command_failed(e.to_string()))?;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| Error::command_failed(e.to_string()))?;
        // Strip the top-level "libtorch/" directory from the archive paths.
        let rel = entry
            .name()
            .trim_start_matches("libtorch/")
            .trim_start_matches('/');
        if rel.is_empty() || entry.is_dir() {
            continue;
        }
        let out = dest.join(rel);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::command_failed(e.to_string()))?;
        }
        let mut f = std::fs::File::create(&out).map_err(|e| Error::command_failed(e.to_string()))?;
        std::io::copy(&mut entry, &mut f).map_err(|e| Error::command_failed(e.to_string()))?;
    }
    Ok(())
}

pub fn download(data_dir: &Path, mut on_progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let backend = detect_backend();
    let url = url(backend)
        .ok_or_else(|| Error::command_failed("no libtorch download URL for this platform".into()))?;
    log::info!("libtorch download ({backend:?}): {url}");

    let temp = data_dir.join("temp");
    std::fs::create_dir_all(&temp).map_err(|e| Error::command_failed(e.to_string()))?;
    let archive = temp.join("libtorch.zip");
    fetch(url, &archive, &mut on_progress)?;

    let dest = libtorch_dir(data_dir);
    std::fs::create_dir_all(&dest).map_err(|e| Error::command_failed(e.to_string()))?;
    extract_zip(&archive, &dest)?;

    let _ = std::fs::remove_file(&archive);
    log::info!("libtorch installed to {}", dest.display());
    Ok(dest)
}
