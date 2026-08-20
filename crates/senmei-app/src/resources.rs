//! Hardware telemetry: CPU/RAM via sysinfo, GPU via `/sys/class/drm` (Linux).

use std::sync::{Mutex, OnceLock};

#[cfg(target_os = "linux")]
use std::path::Path;

use serde::Serialize;
use specta::Type;
use sysinfo::System;

/// Kept alive across samples so CPU usage has a previous sample to diff against.
static SYSTEM: OnceLock<Mutex<System>> = OnceLock::new();

#[derive(Clone, Default, Serialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct HardwareSnapshot {
    /// Overall system CPU load in 0..1.
    pub cpu_usage: f32,
    pub memory_total_bytes: u64,
    pub memory_used_bytes: u64,
    pub gpu_name: Option<String>,
    pub gpu_utilization_percent: Option<f32>,
    pub gpu_memory_used_bytes: Option<u64>,
    pub gpu_memory_total_bytes: Option<u64>,
}

pub fn sample_hardware() -> HardwareSnapshot {
    let mut sys = SYSTEM
        .get_or_init(|| Mutex::new(System::new_all()))
        .lock()
        .unwrap();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let cpu_usage = sys.global_cpu_usage() / 100.0;
    let memory_total_bytes = sys.total_memory();
    let memory_used_bytes = sys.used_memory();
    drop(sys);

    let gpu = sample_gpu();
    HardwareSnapshot {
        cpu_usage,
        memory_total_bytes,
        memory_used_bytes,
        gpu_name: gpu.as_ref().map(|g| g.name.clone()),
        gpu_utilization_percent: gpu.as_ref().and_then(|g| g.utilization),
        gpu_memory_used_bytes: gpu.as_ref().and_then(|g| g.vram_used),
        gpu_memory_total_bytes: gpu.as_ref().and_then(|g| g.vram_total),
    }
}

struct GpuSample {
    name: String,
    utilization: Option<f32>,
    vram_used: Option<u64>,
    vram_total: Option<u64>,
}

/// Primary GPU: the adapter with the most VRAM (discrete over iGPU). None when
/// no card exposes a readable VRAM pair.
#[cfg(target_os = "linux")]
fn sample_gpu() -> Option<GpuSample> {
    let mut best: Option<GpuSample> = None;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let card = entry.file_name().to_string_lossy().into_owned();
        if !card.starts_with("card") || card.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let Some((total, used)) = read_memory_pair(&device) else {
            continue;
        };
        let vendor = read_hex(device.join("vendor")).unwrap_or(0);
        let name = match vendor {
            0x10de => format!("NVIDIA {card}"),
            0x1002 | 0x1022 => format!("AMD {card}"),
            0x8086 => format!("Intel {card}"),
            _ => card,
        };
        let utilization = read_number(device.join("gpu_busy_percent")).map(|v| v as f32);
        if best.as_ref().is_none_or(|b| total > b.vram_total.unwrap_or(0)) {
            best = Some(GpuSample {
                name,
                utilization,
                vram_used: Some(used),
                vram_total: Some(total),
            });
        }
    }
    best
}

#[cfg(not(target_os = "linux"))]
fn sample_gpu() -> Option<GpuSample> {
    None
}

#[cfg(target_os = "linux")]
fn read_memory_pair(device: &Path) -> Option<(u64, u64)> {
    [
        ("mem_info_vram_total", "mem_info_vram_used"),
        ("tile0/vram0/total_bytes", "tile0/vram0/used_bytes"),
    ]
    .into_iter()
    .find_map(|(total, used)| {
        Some((read_number(device.join(total))?, read_number(device.join(used))?))
    })
}

#[cfg(target_os = "linux")]
fn read_number(path: impl AsRef<Path>) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn read_hex(path: impl AsRef<Path>) -> Option<u32> {
    let value = std::fs::read_to_string(path).ok()?;
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}
