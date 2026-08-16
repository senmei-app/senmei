use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{downloader, process};
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



fn libtorch_version(dir: &Path) -> Option<String> {
    let path = dir.join("build-version");
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn driver_version(backend: LibTorchBackend) -> Option<String> {
    match backend {
        LibTorchBackend::Cuda => process::command_output(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        )
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .filter(|s| !s.is_empty()),
        LibTorchBackend::Rocm => {
            process::command_output("rocm-smi", &["--showdriverversion"]).and_then(|s| {
                s.lines()
                    .find(|l| l.to_ascii_lowercase().contains("version"))
                    .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()))
            })
        }
        LibTorchBackend::Cpu => None,
    }
}

fn devices(backend: LibTorchBackend) -> Vec<String> {
    match backend {
        LibTorchBackend::Cuda => process::command_output("nvidia-smi", &["-L"])
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
        LibTorchBackend::Rocm => process::command_output("rocm-smi", &["--showproductname"])
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
    // Must match the libtorch version expected by torch-sys (see senmei-ml).
    // Newer archives use the `libtorch-shared-with-deps` filename (no `cxx11-abi`).
    match backend {
        LibTorchBackend::Cpu => Some(
            "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip",
        ),
        LibTorchBackend::Cuda => Some(
            "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.11.0%2Bcu126.zip",
        ),
        LibTorchBackend::Rocm => Some(
            "https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-2.11.0%2Brocm7.1.zip",
        ),
    }
}

pub fn download(data_dir: &Path, mut on_progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let backend = detect_backend();
    let url = url(backend)
        .ok_or_else(|| Error::Command("no libtorch download URL for this platform".into()))?;
    log::info!("libtorch download ({backend:?}): {url}");

    let archive = downloader::download_to_temp(url, &data_dir.join("temp"), "libtorch.zip", None, &mut on_progress)?;

    let dest = libtorch_dir(data_dir);
    std::fs::create_dir_all(&dest).map_err(Error::from)?;
    downloader::extract_zip(&archive, &dest, "libtorch/")?;

    let _ = std::fs::remove_file(&archive);
    log::info!("libtorch installed to {}", dest.display());
    Ok(dest)
}
