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
pub struct GpuInfo {
    pub name: String,
    pub index: u32,
    pub vram_total_bytes: Option<u64>,
}

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
    /// All detected GPUs for selection.
    pub gpus: Vec<GpuInfo>,
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

    let gpus = enumerate_gpus();
    let primary = gpus.iter().max_by_key(|g| g.vram_total.unwrap_or(0));
    // Read utilization for the primary GPU from sysfs.
    let (gpu_util, gpu_mem_used) = primary.and_then(|g| {
        // Find the matching card entry to read live stats.
        let entries = std::fs::read_dir("/sys/class/drm").ok()?;
        for entry in entries.flatten() {
            let card = entry.file_name().to_string_lossy().into_owned();
            if !card.starts_with("card") || card.contains('-') {
                continue;
            }
            let vendor = read_hex(entry.path().join("device/vendor")).unwrap_or(0);
            let name = match vendor {
                0x10de => format!("NVIDIA {card}"),
                0x1002 | 0x1022 => format!("AMD {card}"),
                0x8086 => format!("Intel {card}"),
                _ => card,
            };
            if name == g.name {
                let device = entry.path().join("device");
                let util = read_number(device.join("gpu_busy_percent")).map(|v| v as f32);
                let mem = read_memory_pair(&device).map(|(_, used)| used);
                return Some((util, mem));
            }
        }
        None
    }).unwrap_or((None, None));
    HardwareSnapshot {
        cpu_usage,
        memory_total_bytes,
        memory_used_bytes,
        gpu_name: primary.map(|g| g.name.clone()),
        gpu_utilization_percent: gpu_util,
        gpu_memory_used_bytes: gpu_mem_used,
        gpu_memory_total_bytes: primary.and_then(|g| g.vram_total),
        gpus: gpus
            .into_iter()
            .enumerate()
            .map(|(i, g)| GpuInfo {
                name: g.name,
                index: i as u32,
                vram_total_bytes: g.vram_total,
            })
            .collect(),
    }
}

struct GpuSample {
    name: String,
    vram_total: Option<u64>,
}

#[cfg(target_os = "linux")]
fn enumerate_gpus() -> Vec<GpuSample> {
    let mut gpus = Vec::new();
    let Some(entries) = std::fs::read_dir("/sys/class/drm").ok() else {
        return gpus;
    };
    for entry in entries.flatten() {
        let card = entry.file_name().to_string_lossy().into_owned();
        if !card.starts_with("card") || card.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let (total, _used) = match read_memory_pair(&device) {
            Some(v) => v,
            None => continue,
        };
        let vendor = read_hex(device.join("vendor")).unwrap_or(0);
        let name = match vendor {
            0x10de => format!("NVIDIA {card}"),
            0x1002 | 0x1022 => format!("AMD {card}"),
            0x8086 => format!("Intel {card}"),
            _ => card,
        };
        gpus.push(GpuSample {
            name,
            vram_total: Some(total),
        });
    }
    gpus
}

#[cfg(not(target_os = "linux"))]
fn enumerate_gpus() -> Vec<GpuSample> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn read_memory_pair(device: &Path) -> Option<(u64, u64)> {
    [
        ("mem_info_vram_total", "mem_info_vram_used"),
        ("tile0/vram0/total_bytes", "tile0/vram0/used_bytes"),
    ]
    .into_iter()
    .find_map(|(total, used)| {
        Some((
            read_number(device.join(total))?,
            read_number(device.join(used))?,
        ))
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
