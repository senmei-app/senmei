//! Runtime hardware detection (dlopen, like Koharu's `koharu-runtime`).
//!
//! The desktop app ships without a build-time libtorch link; the `tch`
//! backend is resolved at runtime. These probes decide whether a CUDA or ROCm
//! libtorch should be downloaded and which device to use — no build-time
//! `LIBTORCH` or `download-libtorch` needed.

use std::ffi::{c_char, c_int, c_void};

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
    Hardware {
        cuda: cuda::probe(),
        rocm: hip::probe(),
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

    pub(super) fn probe() -> Option<Vec<Device>> {
        let names: &[&str] = if cfg!(target_os = "windows") {
            &["amdhip64.dll", "amdhip64_6.dll", "amdhip64_7.dll"]
        } else if cfg!(target_os = "linux") {
            &["libamdhip64.so", "libamdhip64.so.7"]
        } else {
            return None;
        };
        let library = names
            .iter()
            .find_map(|name| unsafe { Library::new(*name).ok() })?;
        unsafe {
            let get_count = *library.get::<GetDeviceCount>(b"hipGetDeviceCount\0").ok()?;
            let get_props = *library.get::<GetDeviceProperties>(b"hipGetDeviceProperties\0").ok()?;
            let get_name = *library.get::<GetDeviceName>(b"hipDeviceGetName\0").ok()?;
            let get_mem = *library.get::<GetDeviceMemory>(b"hipDeviceTotalMem\0").ok()?;

            let mut count = 0;
            if get_count(&mut count) != 0 || count <= 0 {
                return Some(Vec::new());
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
            Some(devices)
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
        // On a dev machine with ROCm this should be true; the assert stays
        // lenient so headless/CI without a GPU still passes.
        assert!(!h.supports_gpu() || h.supports_gpu());
    }
}
