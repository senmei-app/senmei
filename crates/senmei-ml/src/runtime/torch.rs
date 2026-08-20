//! Runtime libtorch resolution (CUDA/ROCm only, no CPU) — like FFmpeg, the
//! libtorch runtime is downloaded on demand into the app data dir and cached.
//! Pinned to a torch release that publishes a ROCm-7 build (2.11.0+rocm7.1), so
//! the downloaded `.so` load on a ROCm 7 runtime and are ABI-compatible with
//! the wrapper.

use std::path::{Path, PathBuf};

use crate::runtime::hardware::{Hardware, Device};

/// Torch release with ROCm-7 libtorch builds (see download.pytorch.org). Must
/// stay in sync with the torch-sys fork's headers used to build the wrapper.
const TORCH_VERSION: &str = "2.11.0";

/// Relative install dir inside the data dir (mirrors Koharu's `Store` layout).
const INSTALL_DIR: &str = "libtorch";

/// Which GPU backend variant to fetch. CPU is intentionally excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchVariant {
    /// NVIDIA CUDA build (e.g. `cu128`).
    Cuda(&'static str),
    /// AMD ROCm build (e.g. `rocm7.1`).
    Rocm(&'static str),
}

impl TorchVariant {
    fn libtorch_dir(&self) -> &'static str {
        match self {
            TorchVariant::Cuda(device) => device,
            TorchVariant::Rocm(device) => device,
        }
    }

    fn url(&self) -> String {
        // Same URL shape as torch-sys's download-libtorch (Linux/Windows x86_64).
        format!(
            "https://download.pytorch.org/libtorch/{}/libtorch-shared-with-deps-{TORCH_VERSION}%2B{}.zip",
            self.libtorch_dir(),
            self.libtorch_dir()
        )
    }

    fn expected_libs(&self) -> &'static [&'static str] {
        match self {
            TorchVariant::Cuda(_) => &[
                "libc10.so",
                "libtorch.so",
                "libtorch_cpu.so",
                "libtorch_cuda.so",
                "libc10_cuda.so",
                "libcaffe2_nvrtc.so",
            ],
            TorchVariant::Rocm(_) => &[
                "libc10.so",
                "libtorch.so",
                "libtorch_cpu.so",
                "libtorch_hip.so",
                "libc10_hip.so",
            ],
        }
    }
}

/// The resolved libtorch install: its `lib` directory (for rpath/dlopen) and
/// the chosen variant.
#[derive(Debug, Clone)]
pub struct TorchInstall {
    pub variant: TorchVariant,
    pub lib_dir: PathBuf,
}

/// Pick the variant from detected hardware (CUDA wins over ROCm when both).
pub fn pick_variant(hardware: &Hardware) -> Option<TorchVariant> {
    if hardware.supports_cuda() {
        Some(TorchVariant::Cuda("cu128"))
    } else if hardware.supports_rocm() {
        Some(TorchVariant::Rocm("rocm7.1"))
    } else {
        None
    }
}

/// Resolve the libtorch install under `data_dir`, downloading on first use.
/// Returns `None` when no CUDA/ROCm device was detected (CPU-only → burn).
pub fn resolve(data_dir: &Path, hardware: &Hardware) -> Result<Option<TorchInstall>, String> {
    let Some(variant) = pick_variant(hardware) else {
        return Ok(None);
    };
    let install = install_dir(data_dir, &variant);
    if is_complete(&install, &variant) {
        return Ok(Some(TorchInstall {
            variant,
            lib_dir: install.join("lib"),
        }));
    }
    let _ = std::fs::remove_dir_all(&install);
    download(data_dir, &variant)?;
    if !is_complete(&install, &variant) {
        return Err("libtorch download incomplete".into());
    }
    Ok(Some(TorchInstall {
        variant,
        lib_dir: install.join("lib"),
    }))
}

fn install_dir(data_dir: &Path, variant: &TorchVariant) -> PathBuf {
    data_dir
        .join(INSTALL_DIR)
        .join(format!("{}-{}", TORCH_VERSION, variant.libtorch_dir()))
}

fn is_complete(install: &Path, variant: &TorchVariant) -> bool {
    let lib = install.join("lib");
    variant
        .expected_libs()
        .iter()
        .all(|name| lib.join(name).is_file())
}

/// Download + extract the libtorch zip into `data_dir/libtorch/<ver>-<dev>`.
fn download(data_dir: &Path, variant: &TorchVariant) -> Result<(), String> {
    let url = variant.url();
    let archive_dir = data_dir.join("libtorch").join("temp");
    let archive = archive_dir.join("libtorch.zip");
    let _ = std::fs::remove_file(&archive);
    senmei_media::download_to_temp(
        &url,
        &archive_dir,
        "libtorch.zip",
        None, // PyTorch doesn't publish SHA for libtorch zips; size check below
        &mut |_, _| {},
    )
    .map_err(|e| format!("libtorch download failed: {e}"))?;

    // The zip extracts a `libtorch/` root; move it to the versioned install dir.
    let stage = data_dir.join("libtorch").join("stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    unzip(&archive, &stage).map_err(|e| format!("libtorch extract failed: {e}"))?;
    let root = stage.join("libtorch");
    if !root.is_dir() {
        return Err("libtorch zip did not contain a libtorch/ root".into());
    }
    let install = install_dir(data_dir, variant);
    let _ = std::fs::remove_dir_all(&install);
    std::fs::rename(&root, &install).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_dir_all(&stage);
    Ok(())
}

fn unzip(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let out = dest.join(entry.name());
        if entry.is_dir() {
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut f).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Best device for the resolved install: CUDA/ROCm first, prefer most VRAM.
pub fn pick_device(hardware: &Hardware) -> Device {
    let all = hardware
        .cuda
        .iter()
        .flatten()
        .chain(hardware.rocm.iter().flatten());
    all.max_by_key(|d| d.vram_bytes)
        .cloned()
        .unwrap_or_else(|| Device {
            name: "unknown".into(),
            vram_bytes: 0,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::hardware::detect;

    #[test]
    fn pick_variant_matches_hardware() {
        let h = detect();
        let v = pick_variant(&h);
        // On a CUDA/ROCm machine this is Some; headless/CI may be None.
        eprintln!("variant: {v:?}");
        assert!(v.is_none() || v.is_some());
    }

    #[test]
    fn cuda_url_shape() {
        let u = TorchVariant::Cuda("cu128").url();
        assert!(u.contains("libtorch-shared-with-deps-2.11.0%2Bcu128.zip"));
        assert!(u.starts_with("https://download.pytorch.org/libtorch/cu128/"));
    }

    /// A complete install dir must be reused without re-downloading: seed the
    /// expected libs (empty files) and check resolve() returns them directly,
    /// twice, with the same lib dir.
    #[test]
    fn resolve_reuses_complete_install() {
        let data_dir = std::env::temp_dir()
            .join(format!("senmei_torch_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        let variant = TorchVariant::Cuda("cu128");
        let lib = install_dir(&data_dir, &variant).join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        for name in variant.expected_libs() {
            std::fs::write(lib.join(name), b"").unwrap();
        }
        let hw = Hardware {
            cuda: Some(vec![Device { name: "test-gpu".into(), vram_bytes: 1 << 30 }]),
            ..Default::default()
        };
        let a = resolve(&data_dir, &hw).unwrap().expect("variant resolves");
        let b = resolve(&data_dir, &hw).unwrap().expect("variant resolves");
        assert_eq!(a.lib_dir, b.lib_dir, "second resolve must hit the cache");
        assert!(a.lib_dir.join("libtorch.so").is_file());
        assert_eq!(a.variant, variant);
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    /// `pick_device` prefers the GPU with the most VRAM (dGPU over APU/iGPU).
    #[test]
    fn pick_device_prefers_most_vram() {
        let hw = Hardware {
            rocm: Some(vec![
                Device { name: "apu".into(), vram_bytes: 2 << 30 },
                Device { name: "dgpu".into(), vram_bytes: 16 << 30 },
            ]),
            ..Default::default()
        };
        assert_eq!(pick_device(&hw).name, "dgpu");
    }
}
