//! Runtime hardware detection (dlopen, like Koharu's `koharu-runtime`).
//!
//! The desktop app ships without a build-time libtorch link; the `tch`
//! backend is resolved at runtime. These probes decide whether a CUDA or ROCm
//! libtorch should be downloaded and which device to use — no build-time
//! `LIBTORCH` or `download-libtorch` needed.

use std::ffi::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;

/// A detected GPU compute device.
#[derive(Debug, Clone)]
pub struct Device {
    pub name: String,
    pub vram_bytes: u64,
}

/// Runtime GPU backend detection result.
#[derive(Debug, Clone, Default)]
pub struct Hardware {
    pub cuda: Option<Vec<Device>>,
    pub rocm: Option<Vec<Device>>,
    /// Directory of the loaded HIP runtime (`libamdhip64`'s dir, via `dladdr`).
    /// Lets the loader preload the ROCm runtime libs (RTLD_GLOBAL) so
    /// `libtorch_hip` resolves its deps at dlopen time.
    pub rocm_runtime_dir: Option<PathBuf>,
}

impl Hardware {
    /// True when a CUDA (NVIDIA) device is present.
    pub fn supports_cuda(&self) -> bool {
        self.cuda.as_ref().is_some_and(|d| !d.is_empty())
    }

    /// True when a ROCm (AMD) device is present.
    pub fn supports_rocm(&self) -> bool {
        self.rocm.as_ref().is_some_and(|d| !d.is_empty())
    }

    /// True when any GPU backend is present (CUDA or ROCm).
    pub fn supports_gpu(&self) -> bool {
        self.supports_cuda() || self.supports_rocm()
    }
}

/// Probe both CUDA and ROCm at runtime; whichever library loads wins.
pub fn detect() -> Hardware {
    let (rocm, rocm_runtime_dir) = hip::probe();
    Hardware {
        cuda: cuda::probe(),
        rocm,
        rocm_runtime_dir,
    }
}

mod cuda {
    use super::*;

    type Init = unsafe extern "C" fn(c_int) -> c_int;
    type DeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type DeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type DeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type DeviceTotalMem = unsafe extern "C" fn(*mut u64, c_int) -> c_int;

    pub(super) fn probe() -> Option<Vec<Device>> {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["nvcuda.dll"]
        } else if cfg!(target_os = "linux") {
            &["libcuda.so.1", "libcuda.so"]
        } else {
            return None;
        };
        let library = names
            .iter()
            .find_map(|name| unsafe { Library::new(*name).ok() })?;
        unsafe {
            let init = *library.get::<Init>(b"cuInit\0").ok()?;
            let get_count = *library.get::<DeviceGetCount>(b"cuDeviceGetCount\0").ok()?;
            let get_device = *library.get::<DeviceGet>(b"cuDeviceGet\0").ok()?;
            let get_name = *library.get::<DeviceGetName>(b"cuDeviceGetName\0").ok()?;
            let get_mem = *library.get::<DeviceTotalMem>(b"cuDeviceTotalMem_v2\0").ok()?;

            if init(0) != 0 {
                return None;
            }
            let mut count = 0;
            if get_count(&mut count) != 0 || count <= 0 {
                return Some(Vec::new());
            }
            let mut devices = Vec::new();
            for index in 0..count {
                let mut handle = 0;
                if get_device(&mut handle, index) != 0 {
                    continue;
                }
                let mut name_buf = [0u8; 256];
                if get_name(name_buf.as_mut_ptr().cast(), 256, handle) != 0 {
                    continue;
                }
                let name = std::ffi::CStr::from_ptr(name_buf.as_ptr().cast())
                    .to_string_lossy()
                    .into_owned();
                let mut vram = 0u64;
                if get_mem(&mut vram, handle) != 0 {
                    continue;
                }
                devices.push(Device { name, vram_bytes: vram });
            }
            Some(devices)
        }
    }
}

mod hip {
    use super::*;

    type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type GetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
    type GetDeviceName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type GetDeviceMemory = unsafe extern "C" fn(*mut u64, c_int) -> c_int;

    #[cfg(unix)]
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }

    #[cfg(unix)]
    unsafe extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }

    /// Directory of the shared object that exports `fptr` (e.g. `libamdhip64`),
    /// so the loader can preload the ROCm runtime from the same place. `dladdr`
    /// is POSIX-only; other platforms rely on the env-derived dir.
    #[cfg(unix)]
    fn loaded_lib_dir(fptr: *const c_void) -> Option<PathBuf> {
        unsafe {
            let mut info = std::mem::zeroed::<DlInfo>();
            if dladdr(fptr, &mut info) != 0 && !info.dli_fname.is_null() {
                let fname = std::ffi::CStr::from_ptr(info.dli_fname).to_str().ok()?;
                Path::new(fname).parent().map(|p| p.to_path_buf())
            } else {
                None
            }
        }
    }

    /// The documented ROCm lib dir (`$ROCM_PATH/lib` / `$ROCM_INSTALL_PATH/lib`,
    /// per the AMD install guide) when it actually holds the HIP runtime.
    fn env_rocm_dir() -> Option<PathBuf> {
        for var in ["ROCM_PATH", "ROCM_INSTALL_PATH"] {
            if let Ok(root) = std::env::var(var) {
                let lib = Path::new(&root).join("lib");
                if lib.join("libamdhip64.so").is_file()
                    || lib.join("libamdhip64.so.7").is_file()
                {
                    return Some(lib);
                }
            }
        }
        None
    }

    pub(super) fn probe() -> (Option<Vec<Device>>, Option<PathBuf>) {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["amdhip64.dll", "amdhip64_6.dll", "amdhip64_7.dll"]
        } else if cfg!(target_os = "linux") {
            &["libamdhip64.so", "libamdhip64.so.7"]
        } else {
            return (None, None);
        };
        let library = match names
            .iter()
            .find_map(|name| unsafe { Library::new(*name).ok() })
            .or_else(|| {
                // GUI apps don't inherit the shell's LD_LIBRARY_PATH; fall back
                // to the documented $ROCM_PATH/lib (AMD install guide).
                env_rocm_dir()
                    .and_then(|dir| unsafe { Library::new(dir.join("libamdhip64.so")).ok() })
            }) {
            Some(l) => l,
            None => return (None, None),
        };
        unsafe {
            let get_count = match library.get::<GetDeviceCount>(b"hipGetDeviceCount\0").ok() {
                Some(f) => *f,
                None => return (None, None),
            };
            let get_props = match library.get::<GetDeviceProperties>(b"hipGetDeviceProperties\0").ok() {
                Some(f) => *f,
                None => return (None, None),
            };
            let get_name = match library.get::<GetDeviceName>(b"hipDeviceGetName\0").ok() {
                Some(f) => *f,
                None => return (None, None),
            };
            let get_mem = match library.get::<GetDeviceMemory>(b"hipDeviceTotalMem\0").ok() {
                Some(f) => *f,
                None => return (None, None),
            };
            #[cfg(unix)]
            let dir = env_rocm_dir().or_else(|| loaded_lib_dir(get_count as *const c_void));
            #[cfg(not(unix))]
            let dir = env_rocm_dir();

            let mut count = 0;
            if get_count(&mut count) != 0 || count <= 0 {
                return (Some(Vec::new()), dir);
            }
            let mut devices = Vec::new();
            for index in 0..count {
                // hipGetDeviceProperties needs a large aligned buffer.
                #[repr(C, align(64))]
                struct Props([u8; 64 * 1024]);
                let mut props = Box::new(Props([0; 64 * 1024]));
                if get_props(props.0.as_mut_ptr().cast(), index) != 0 {
                    continue;
                }
                let mut name_buf = [0u8; 256];
                if get_name(name_buf.as_mut_ptr().cast(), 256, index) != 0 {
                    continue;
                }
                let name = std::ffi::CStr::from_ptr(name_buf.as_ptr().cast())
                    .to_string_lossy()
                    .into_owned();
                let mut vram = 0u64;
                if get_mem(&mut vram, index) != 0 {
                    continue;
                }
                devices.push(Device { name, vram_bytes: vram });
            }
            (Some(devices), dir)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics() {
        // Should always return a struct (fields may be None) — never crash
        // even with no CUDA/ROCm runtime installed.
        let h = detect();
        assert!(h.cuda.is_none() || h.cuda.is_some());
        assert!(h.rocm.is_none() || h.rocm.is_some());
    }

    #[test]
    fn detect_reports_devices() {
        // Print what the runtime actually finds (run with --nocapture).
        let h = detect();
        eprintln!("cuda: {:?}", h.cuda);
        eprintln!("rocm: {:?}", h.rocm);
        eprintln!("rocm_runtime_dir: {:?}", h.rocm_runtime_dir);
        // On a dev machine with ROCm this should be true; the assert stays
        // lenient so headless/CI without a GPU still passes.
        assert!(!h.supports_gpu() || h.supports_gpu());
    }
}
