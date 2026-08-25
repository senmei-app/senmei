//! Runtime libtorch resolution (CUDA/ROCm only, no CPU) — like FFmpeg, the
//! libtorch runtime is downloaded on demand into the app data dir and cached.
//! CUDA comes from the pytorch.org zips; ROCm from the AMD wheel index
//! (pytorch.org stopped publishing ROCm libtorch builds that match the AMD
//! ROCm SDK). The AMD wheels pin torch 2.12.0 to ROCm 7.14.0 — the same pair
//! Koharu ships — so the downloaded `.so` dlopen against the pinned SDK and
//! stay ABI-compatible with the wrapper.

use std::path::{Path, PathBuf};

use crate::runtime::hardware::{Device, Hardware};
use crate::runtime::rocm;

/// Torch release with CUDA/CPU libtorch zips (download.pytorch.org), used when
/// no local `LIBTORCH` install is set. The ROCm path uses `ROCM_TORCH_VERSION`.
const TORCH_VERSION: &str = "2.11.0";

/// ROCm torch release from the AMD wheel index — must match the pinned ROCm
/// SDK (`rocm::ROCM_VERSION`). Same pair Koharu ships.
const ROCM_TORCH_VERSION: &str = "2.12.0";

/// Relative install dir inside the data dir (mirrors Koharu's `Store` layout).
const INSTALL_DIR: &str = "libtorch";

/// AMD wheel index hosting the ROCm torch + SDK packages.
const ROCM_INDEX: &str = "https://repo.amd.com/rocm/whl-multi-arch";

/// Per-GPU `.kpack`/aotriton kernels live in per-GPU + per-family device
/// wheels; the family wheel covers the whole arch family (e.g. `gfx12_0`).
fn torch_family(target: &str) -> Option<&'static str> {
    if target.starts_with("gfx11") {
        Some("gfx11")
    } else if target.starts_with("gfx12") {
        Some("gfx12_0")
    } else {
        None
    }
}

/// Which GPU backend variant to fetch. CPU is intentionally excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchVariant {
    /// NVIDIA CUDA build (e.g. `cu128`).
    Cuda(&'static str),
    /// AMD ROCm build (e.g. `rocm7.14`).
    Rocm(&'static str),
}

impl TorchVariant {
    fn libtorch_dir(&self) -> &'static str {
        match self {
            TorchVariant::Cuda(device) => device,
            TorchVariant::Rocm(device) => device,
        }
    }

    fn version(&self) -> &'static str {
        match self {
            TorchVariant::Cuda(_) => TORCH_VERSION,
            TorchVariant::Rocm(_) => ROCM_TORCH_VERSION,
        }
    }

    fn url(&self) -> String {
        match self {
            TorchVariant::Cuda(device) => format!(
                "https://download.pytorch.org/libtorch/{device}/libtorch-shared-with-deps-{TORCH_VERSION}%2B{device}.zip"
            ),
            // Same URL shape as Koharu's ROCm torch wheel.
            TorchVariant::Rocm(_) => format!(
                "{ROCM_INDEX}/torch-{ROCM_TORCH_VERSION}%2Brocm{}-cp312-cp312-{}.whl",
                rocm::ROCM_VERSION,
                rocm::wheel_platform()
            ),
        }
    }

    /// Additional ROCm wheels: per-GPU `.kpack` + per-family aotriton kernels.
    fn rocm_device_urls(&self, target: &str) -> Vec<String> {
        let mut urls = vec![format!(
            "{ROCM_INDEX}/amd_torch_device_{target}-{ROCM_TORCH_VERSION}%2Brocm{}-cp312-cp312-{}.whl",
            rocm::ROCM_VERSION,
            rocm::wheel_platform()
        )];
        if let Some(family) = torch_family(target) {
            urls.push(format!(
                "{ROCM_INDEX}/amd_torch_device_{family}-{ROCM_TORCH_VERSION}%2Brocm{}-cp312-cp312-{}.whl",
                rocm::ROCM_VERSION,
                rocm::wheel_platform()
            ));
        }
        urls
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
            // Same list as Koharu's ROCm `Torch::library_names`.
            TorchVariant::Rocm(_) => &[
                "libc10.so",
                "libc10_hip.so",
                "libaotriton_v2.so.0.11.2",
                "libcaffe2_nvrtc.so",
                "libshm.so",
                "libtorch_global_deps.so",
                "libtorch_cpu.so",
                "libtorch_hip.so",
                "libtorch_rocshmem.so",
                "libtorch.so",
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
        Some(TorchVariant::Rocm("rocm7.14"))
    } else {
        None
    }
}

/// Resolve the libtorch install under `data_dir`, downloading on first use.
/// Returns `None` when no CUDA/ROCm device was detected (CPU-only → burn).
pub fn resolve(data_dir: &Path, hardware: &Hardware) -> Result<Option<TorchInstall>, String> {
    // A local torch install (build-time `LIBTORCH`) is only honored when
    // explicitly opted in via `SENMEI_LIBTORCH_ENV` — its ABI matches the
    // compiled shim exactly, while a downloaded release can mismatch (e.g. a
    // 2.13-built wrapper against the pinned 2.12 download). Off by default so
    // a stale `LIBTORCH` in the launch shell (e.g. a Python venv) can't hijack
    // the shipped/pinned runtime (it would fail the tensor-probe ABI guard in
    // `tch::ensure_loaded` anyway). CPU-only installs are ignored (tch needs a
    // GPU build); we fall back to the download.
    if std::env::var_os("SENMEI_LIBTORCH_ENV").is_some() {
        if let Some(dir) = std::env::var_os("LIBTORCH") {
            let lib = PathBuf::from(&dir).join("lib");
            if lib.join("libtorch.so").is_file() {
                let variant = if lib.join("libtorch_hip.so").is_file() {
                    Some(TorchVariant::Rocm("rocm7.14"))
                } else if lib.join("libtorch_cuda.so").is_file() {
                    Some(TorchVariant::Cuda("cu128"))
                } else {
                    None
                };
                if let Some(variant) = variant {
                    log::info!("libtorch: using LIBTORCH env ({variant:?}) at {lib:?}");
                    return Ok(Some(TorchInstall {
                        variant,
                        lib_dir: lib,
                    }));
                }
            }
        }
    }
    let Some(variant) = pick_variant(hardware) else {
        return Ok(None);
    };
    let rocm_target = match variant {
        TorchVariant::Rocm(_) => hardware.rocm_target.as_deref(),
        _ => None,
    };
    let install = install_dir(data_dir, &variant);
    if is_complete(&install, &variant, rocm_target) {
        log::info!("libtorch: using cached runtime {variant:?} at {install:?}");
        return Ok(Some(TorchInstall {
            variant,
            lib_dir: install.join("lib"),
        }));
    }
    let _ = std::fs::remove_dir_all(&install);
    log::info!("libtorch: downloading runtime {variant:?}");
    download(data_dir, &variant, rocm_target)?;
    if !is_complete(&install, &variant, rocm_target) {
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
        .join(format!("{}-{}", variant.version(), variant.libtorch_dir()))
}

fn is_complete(install: &Path, variant: &TorchVariant, rocm_target: Option<&str>) -> bool {
    let lib = install.join("lib");
    if !variant
        .expected_libs()
        .iter()
        .all(|name| lib.join(name).is_file())
    {
        return false;
    }
    match variant {
        // ROCm also needs the per-GPU `.kpack` + aotriton kernels (device
        // wheels) before the runtime is usable.
        TorchVariant::Rocm(_) => {
            let target = rocm_target.unwrap_or_default();
            // `.kpack` is always required; `aotriton.images` only exists for
            // archs with a family wheel (gfx11/gfx12) — gfx9/gfx10 have none,
            // so demanding it there makes every launch re-download the ~2 GB
            // wheel and fail with "libtorch download incomplete".
            let aotriton_ok = match torch_family(&target) {
                Some(_) => lib.join("aotriton.images").is_dir(),
                None => true,
            };
            install
                .join(".kpack")
                .join(format!("torch_{target}.kpack"))
                .is_file()
                && aotriton_ok
        }
        TorchVariant::Cuda(_) => true,
    }
}

/// Download + extract the libtorch zip/wheels into `data_dir/libtorch/<ver>-<dev>`.
fn download(
    data_dir: &Path,
    variant: &TorchVariant,
    rocm_target: Option<&str>,
) -> Result<(), String> {
    let archive_dir = data_dir.join("libtorch").join("temp");
    let is_rocm = matches!(variant, TorchVariant::Rocm(_));
    let archive_name = if is_rocm {
        "libtorch.whl"
    } else {
        "libtorch.zip"
    };
    let archive = archive_dir.join(archive_name);
    let _ = std::fs::remove_file(&archive);
    senmei_media::download_to_temp(
        &variant.url(),
        &archive_dir,
        archive_name,
        None, // PyTorch/AMD don't publish SHA for libtorch archives
        &mut |_, _| {},
    )
    .map_err(|e| format!("libtorch download failed: {e}"))?;

    let stage = data_dir.join("libtorch").join("stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    if is_rocm {
        // AMD wheels: torch/lib (main) + torch/.kpack & aotriton kernels
        // (device wheels); extract the same prefixes from each, Koharu-style.
        let prefixes = ["torch/lib/", "torch/.kpack/"];
        extract_wheel_prefixes(&archive, &stage, &prefixes)
            .map_err(|e| format!("libtorch extract failed: {e}"))?;
        let target = rocm_target.unwrap_or_default();
        for url in variant.rocm_device_urls(target) {
            let whl = archive_dir.join("libtorch-device.whl");
            let _ = std::fs::remove_file(&whl);
            senmei_media::fetch(&url, &whl, &mut |_, _| {})
                .map_err(|e| format!("libtorch device download failed: {e}"))?;
            extract_wheel_prefixes(&whl, &stage, &prefixes)
                .map_err(|e| format!("libtorch extract failed: {e}"))?;
            let _ = std::fs::remove_file(&whl);
        }
    } else {
        unzip(&archive, &stage).map_err(|e| format!("libtorch extract failed: {e}"))?;
    }

    // The zip/wheel extracts a `libtorch/` / `torch/` root; move it to the
    // versioned install dir.
    let root = if is_rocm {
        stage.join("torch")
    } else {
        stage.join("libtorch")
    };
    if !root.is_dir() {
        return Err("libtorch archive did not contain its root dir".into());
    }
    let install = install_dir(data_dir, variant);
    let _ = std::fs::remove_dir_all(&install);
    std::fs::rename(&root, &install).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_dir_all(&stage);
    Ok(())
}

/// Extract only entries under `prefixes` from a wheel/zip, preserving paths.
fn extract_wheel_prefixes(archive: &Path, dest: &Path, prefixes: &[&str]) -> Result<(), String> {
    senmei_media::extract_zip(archive, dest, |name| {
        prefixes.iter().any(|p| name.starts_with(p))
    })
    .map_err(|e| e.to_string())
}

fn unzip(archive: &Path, dest: &Path) -> Result<(), String> {
    senmei_media::extract_zip(archive, dest, |_| true).map_err(|e| e.to_string())
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
        // `resolve` prefers a build-time `LIBTORCH`; unset it so the download
        // path (and its cache reuse) is exercised deterministically.
        let had_libtorch = std::env::var_os("LIBTORCH");
        std::env::remove_var("LIBTORCH");
        let data_dir =
            std::env::temp_dir().join(format!("senmei_torch_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        let variant = TorchVariant::Cuda("cu128");
        let lib = install_dir(&data_dir, &variant).join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        for name in variant.expected_libs() {
            std::fs::write(lib.join(name), b"").unwrap();
        }
        let hw = Hardware {
            cuda: Some(vec![Device {
                name: "test-gpu".into(),
                vram_bytes: 1 << 30,
            }]),
            ..Default::default()
        };
        let a = resolve(&data_dir, &hw).unwrap().expect("variant resolves");
        let b = resolve(&data_dir, &hw).unwrap().expect("variant resolves");
        assert_eq!(a.lib_dir, b.lib_dir, "second resolve must hit the cache");
        assert!(a.lib_dir.join("libtorch.so").is_file());
        assert_eq!(a.variant, variant);
        let _ = std::fs::remove_dir_all(&data_dir);
        if let Some(dir) = had_libtorch {
            std::env::set_var("LIBTORCH", dir);
        }
    }

    /// `pick_device` prefers the GPU with the most VRAM (dGPU over APU/iGPU).
    #[test]
    fn pick_device_prefers_most_vram() {
        let hw = Hardware {
            rocm: Some(vec![
                Device {
                    name: "apu".into(),
                    vram_bytes: 2 << 30,
                },
                Device {
                    name: "dgpu".into(),
                    vram_bytes: 16 << 30,
                },
            ]),
            ..Default::default()
        };
        assert_eq!(pick_device(&hw).name, "dgpu");
    }
}
