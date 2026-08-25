//! Runtime hardware detection (dlopen, like Koharu's `koharu-runtime`).
//!
//! The desktop app ships without a build-time libtorch link; the `tch`
//! backend is resolved at runtime. These probes decide whether a CUDA or ROCm
//! libtorch should be downloaded and which device to use — no build-time
//! `LIBTORCH` or `download-libtorch` needed.

use std::ffi::{c_char, c_int};

use libloading::Library;

// `c_void` is only used by the non-Linux HIP dlopen probe.
#[cfg(not(target_os = "linux"))]
use std::ffi::c_void;

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
    /// Per-GPU ROCm target (e.g. `gfx1201`), for the on-demand SDK download.
    pub rocm_target: Option<String>,
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
    let (rocm, rocm_target) = hip::probe();
    Hardware {
        cuda: cuda::probe(),
        rocm,
        rocm_target,
    }
}

/// `(total, used)` VRAM of the discrete GPU with the most memory (Linux DRM
/// sysfs), for the fused-path VRAM guard. `None` when unreadable.
#[cfg(target_os = "linux")]
fn vram_mem_info() -> Option<(u64, u64)> {
    let mut best: Option<(u64, u64)> = None;
    let entries = std::fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("renderD") {
            continue;
        }
        let base = entry.path().join("device");
        let read = |f: &str| {
            std::fs::read_to_string(base.join(f))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
        };
        let total = read("mem_info_vram_total")?;
        let used = read("mem_info_vram_used").unwrap_or(0);
        if best.map_or(true, |(t, _)| total > t) {
            best = Some((total, used));
        }
    }
    best
}

/// Free VRAM on the discrete GPU with the most memory (`total − used`), for
/// the fused-path VRAM guard. `None` when unreadable.
#[cfg(target_os = "linux")]
pub fn vram_available_bytes() -> Option<u64> {
    vram_mem_info().map(|(t, u)| t.saturating_sub(u))
}

/// Total VRAM of the discrete GPU with the most memory, for the fused-path
/// VRAM guard's system-adaptive ceiling. `None` when unreadable.
#[cfg(target_os = "linux")]
pub fn vram_total_bytes() -> Option<u64> {
    vram_mem_info().map(|(t, _)| t)
}

#[cfg(not(target_os = "linux"))]
pub fn vram_available_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "linux"))]
pub fn vram_total_bytes() -> Option<u64> {
    None
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
            let get_mem = *library
                .get::<DeviceTotalMem>(b"cuDeviceTotalMem_v2\0")
                .ok()?;

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
                devices.push(Device {
                    name,
                    vram_bytes: vram,
                });
            }
            Some(devices)
        }
    }
}

mod hip {
    use super::*;

    // Linux: read the kernel KFD topology — never initializes a HIP runtime.
    // Probing the system HIP first would poison the SDK HIP that torch later
    // preloads: HSA's runtime state is shared with the kernel driver, so a
    // second, different HIP runtime can't attach to it and reports
    // "No CUDA GPUs are available" (the render then crashes once it moves
    // tensors to the GPU).
    #[cfg(target_os = "linux")]
    pub(super) fn probe() -> (Option<Vec<Device>>, Option<String>) {
        const NODES: &str = "/sys/class/kfd/kfd/topology/nodes";
        let Ok(nodes) = std::fs::read_dir(NODES) else {
            return (None, None);
        };
        let mut found: Vec<(String, u64)> = Vec::new();
        for entry in nodes.flatten() {
            let Ok(props) = std::fs::read_to_string(entry.path().join("properties")) else {
                continue;
            };
            // `gfx_target_version` is hex `major<<16 | minor<<8 | stepping`,
            // e.g. `120001` → gfx1201 (gfx{major:x}{minor:x}{stepping:x}).
            let Some(v) = props
                .lines()
                .find_map(|l| l.strip_prefix("gfx_target_version "))
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            else {
                continue; // CPU node
            };
            let major = (v >> 16) & 0xff;
            let minor = (v >> 8) & 0xff;
            let stepping = v & 0xff;
            if major == 0 {
                continue; // CPU node
            }
            let target = format!("gfx{major:x}{minor:x}{stepping:x}");
            let vram = props
                .lines()
                .find_map(|l| l.strip_prefix("drm_render_minor "))
                .and_then(|s| s.trim().parse::<u32>().ok())
                .and_then(|minor| {
                    std::fs::read_to_string(format!(
                        "/sys/class/drm/renderD{minor}/device/mem_info_vram_total"
                    ))
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                })
                .unwrap_or(0);
            found.push((target, vram));
        }
        if found.is_empty() {
            return (None, None);
        }
        let devices = found
            .iter()
            .map(|(t, v)| Device {
                name: format!("AMD Radeon {t}"),
                vram_bytes: *v,
            })
            .collect();
        // The tch engine uses device 0 = most VRAM (dGPU over iGPU); its gfx
        // target drives the on-demand SDK download.
        let target = found.iter().max_by_key(|(_, v)| *v).map(|(t, _)| t.clone());
        (Some(devices), target)
    }

    // non-Linux: dlopen HIP directly (no shared HSA driver state to poison).
    #[cfg(not(target_os = "linux"))]
    pub(super) fn probe() -> (Option<Vec<Device>>, Option<String>) {
        type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
        type GetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
        type GetDeviceName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
        type GetDeviceMemory = unsafe extern "C" fn(*mut u64, c_int) -> c_int;

        let names: &[&str] = if cfg!(target_os = "windows") {
            &["amdhip64.dll", "amdhip64_6.dll", "amdhip64_7.dll"]
        } else {
            &["libamdhip64.so", "libamdhip64.so.7"]
        };
        let Some(library) = names
            .iter()
            .find_map(|name| unsafe { Library::new(*name).ok() })
        else {
            return (None, None);
        };
        unsafe {
            let get_count = match library.get::<GetDeviceCount>(b"hipGetDeviceCount\0") {
                Ok(f) => *f,
                Err(_) => return (None, None),
            };
            let get_props = match library.get::<GetDeviceProperties>(b"hipGetDeviceProperties\0") {
                Ok(f) => *f,
                Err(_) => return (None, None),
            };
            let get_name = match library.get::<GetDeviceName>(b"hipDeviceGetName\0") {
                Ok(f) => *f,
                Err(_) => return (None, None),
            };
            let get_mem = match library.get::<GetDeviceMemory>(b"hipDeviceTotalMem\0") {
                Ok(f) => *f,
                Err(_) => return (None, None),
            };
            let mut count = 0;
            if get_count(&mut count) != 0 || count <= 0 {
                return (Some(Vec::new()), None);
            }
            let mut devices = Vec::new();
            let mut gfx_target = None;
            for index in 0..count {
                #[repr(C, align(64))]
                struct Props([u8; 64 * 1024]);
                let mut props = Box::new(Props([0; 64 * 1024]));
                if get_props(props.0.as_mut_ptr().cast(), index) != 0 {
                    continue;
                }
                if gfx_target.is_none() {
                    gfx_target = target(&props.0);
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
                devices.push(Device {
                    name,
                    vram_bytes: vram,
                });
            }
            (Some(devices), gfx_target)
        }
    }

    /// The per-GPU target string (e.g. `gfx1201`) inside the `hipDeviceProp_t`
    /// struct (`gcnArchName`), or `None`. Mirrors Koharu's scan.
    #[cfg(not(target_os = "linux"))]
    fn target(properties: &[u8]) -> Option<String> {
        properties
            .windows(3)
            .enumerate()
            .find_map(|(start, bytes)| {
                if bytes != b"gfx" {
                    return None;
                }
                let suffix = properties[start + 3..]
                    .iter()
                    .take_while(|b| b.is_ascii_alphanumeric())
                    .count();
                let t = std::str::from_utf8(&properties[start..start + 3 + suffix]).ok()?;
                t[3..]
                    .bytes()
                    .any(|b| b.is_ascii_digit())
                    .then(|| t.to_owned())
            })
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
        eprintln!("rocm_target: {:?}", h.rocm_target);
        // On a dev machine with ROCm this should be true; the assert stays
        // lenient so headless/CI without a GPU still passes.
        assert!(!h.supports_gpu() || h.supports_gpu());
    }
}
