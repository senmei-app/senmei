//! On-demand per-GPU ROCm SDK runtime (Koharu-style). Downloaded only when the
//! system ROCm runtime can't satisfy the pinned libtorch (missing versioned
//! SONAME libs such as `libMIOpen.so.1` / `librocprofiler-sdk.so.1` /
//! `libamdhip64.so.7`). The pytorch ROCm libtorch zip ships unversioned copies
//! of most ROCm libs but not the versioned names the `.so` files dlopen, so a
//! bare system without ROCm loads libtorch incompletely and the wrapper's
//! tensor ABI breaks (see tch/mod.rs).

#[cfg(feature = "tch")]
use std::path::{Path, PathBuf};

/// ROCm SDK release published on the AMD wheel index; must match the pinned
/// libtorch (`rocm7.14`) — mirrors Koharu's runtime.
pub const ROCM_VERSION: &str = "7.14.0";
#[cfg(feature = "tch")]
pub(crate) const INDEX: &str = "https://repo.amd.com/rocm/whl-multi-arch";

pub(crate) fn wheel_platform() -> &'static str {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "win_amd64"
    } else {
        "linux_x86_64"
    }
}

#[cfg(feature = "tch")]
fn install_dir(data_dir: &Path, target: &str) -> PathBuf {
    data_dir.join("rocm").join(ROCM_VERSION).join(target)
}

/// Whether the per-GPU SDK is fully present on disk.
#[cfg(feature = "tch")]
pub fn is_complete(data_dir: &Path, target: &str) -> bool {
    let root = install_dir(data_dir, target);
    if cfg!(target_os = "windows") {
        root.join("_rocm_sdk_core/bin/amdhip64_7.dll").is_file()
            && root.join("_rocm_sdk_libraries/bin/MIOpen.dll").is_file()
    } else {
        root.join("_rocm_sdk_core/lib/libamdhip64.so.7").is_file()
            && root
                .join("_rocm_sdk_libraries/lib/libMIOpen.so.1")
                .is_file()
    }
}

/// Download + extract the three per-GPU ROCm SDK wheels into
/// `data_dir/rocm/<ver>/<gfx>`; returns the install root. No-op when complete.
#[cfg(feature = "tch")]
pub fn download(data_dir: &Path, target: &str) -> Result<PathBuf, String> {
    let platform = wheel_platform();
    let root = install_dir(data_dir, target);
    if is_complete(data_dir, target) {
        return Ok(root);
    }
    let _ = std::fs::remove_dir_all(&root);
    let stage = data_dir.join("rocm").join("stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    for (whl, prefix) in [
        (
            format!("rocm_sdk_core-{ROCM_VERSION}-py3-none-{platform}.whl"),
            "_rocm_sdk_core",
        ),
        (
            format!("rocm_sdk_libraries-{ROCM_VERSION}-py3-none-{platform}.whl"),
            "_rocm_sdk_libraries",
        ),
        (
            format!("rocm_sdk_device_{target}-{ROCM_VERSION}-py3-none-{platform}.whl"),
            "_rocm_sdk_libraries",
        ),
    ] {
        let url = format!("{INDEX}/{whl}");
        let archive = stage.join(&whl);
        log::info!("rocm sdk: downloading {url}");
        senmei_media::fetch(&url, &archive, &mut |_, _| {})
            .map_err(|e| format!("rocm sdk download failed ({whl}): {e}"))?;
        senmei_media::extract_zip_prefix(&archive, &root, prefix)
            .map_err(|e| format!("rocm sdk extract failed ({whl}): {e}"))?;
        let _ = std::fs::remove_file(&archive);
    }
    let _ = std::fs::remove_dir_all(&stage);
    if !is_complete(data_dir, target) {
        return Err("rocm sdk download incomplete".into());
    }
    Ok(root)
}

/// The SDK library files (relative to the install root) to preload with
/// RTLD_GLOBAL before libtorch, in dependency order — mirrors Koharu's
/// `Rocm::activate`.
#[cfg(feature = "tch")]
pub fn preload_libs() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &[
            "_rocm_sdk_core/bin/amd_comgr.dll",
            "_rocm_sdk_core/bin/rocm_kpack.dll",
            "_rocm_sdk_core/bin/rocm-openblas.dll",
            "_rocm_sdk_core/bin/amdhip64_7.dll",
            "_rocm_sdk_core/bin/hiprtc-builtins0714.dll",
            "_rocm_sdk_core/bin/hiprtc0714.dll",
            "_rocm_sdk_libraries/bin/rocrand.dll",
            "_rocm_sdk_libraries/bin/hiprand.dll",
            "_rocm_sdk_libraries/bin/rocblas.dll",
            "_rocm_sdk_libraries/bin/hipblas.dll",
            "_rocm_sdk_libraries/bin/libhipblaslt.dll",
            "_rocm_sdk_libraries/bin/rocfft.dll",
            "_rocm_sdk_libraries/bin/hipfft.dll",
            "_rocm_sdk_libraries/bin/rocsolver.dll",
            "_rocm_sdk_libraries/bin/hipsolver.dll",
            "_rocm_sdk_libraries/bin/rocsparse.dll",
            "_rocm_sdk_libraries/bin/hipsparse.dll",
            "_rocm_sdk_libraries/bin/MIOpen.dll",
        ]
    } else {
        &[
            "_rocm_sdk_core/lib/librocprofiler-register.so.0",
            "_rocm_sdk_core/lib/libamd_comgr.so.3",
            "_rocm_sdk_core/lib/libhsa-runtime64.so.1",
            "_rocm_sdk_core/lib/libamdhip64.so.7",
            "_rocm_sdk_core/lib/librocprofiler-sdk.so.1",
            "_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1",
            "_rocm_sdk_core/lib/libroctracer64.so.4",
            "_rocm_sdk_core/lib/libroctx64.so.4",
            "_rocm_sdk_core/lib/libhiprtc-builtins.so.7",
            "_rocm_sdk_core/lib/libhiprtc.so.7",
            "_rocm_sdk_core/lib/rocm_sysdeps/lib/librocm_sysdeps_liblzma.so.5",
            "_rocm_sdk_core/lib/host-math/lib/librocm-openblas.so.0",
            "_rocm_sdk_core/lib/librocm_smi64.so.1",
            "_rocm_sdk_libraries/lib/librocblas.so.5",
            "_rocm_sdk_libraries/lib/libhipblas.so.3",
            "_rocm_sdk_libraries/lib/libhipblaslt.so.1",
            "_rocm_sdk_libraries/lib/librocfft.so.0",
            "_rocm_sdk_libraries/lib/libhipfft.so.0",
            "_rocm_sdk_libraries/lib/librocrand.so.1",
            "_rocm_sdk_libraries/lib/libhiprand.so.1",
            "_rocm_sdk_libraries/lib/librocsolver.so.0",
            "_rocm_sdk_libraries/lib/libhipsolver.so.1",
            "_rocm_sdk_libraries/lib/librocsparse.so.1",
            "_rocm_sdk_libraries/lib/libhipsparse.so.4",
            "_rocm_sdk_libraries/lib/libhipsparselt.so.0",
            "_rocm_sdk_libraries/lib/libMIOpen.so.1",
            "_rocm_sdk_libraries/lib/libhipdnn_backend.so",
            "_rocm_sdk_libraries/lib/librccl.so.1",
        ]
    }
}
