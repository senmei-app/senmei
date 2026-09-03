//! Encoder selection: pick the best video codec from ffmpeg's capabilities,
//! verify hardware encoders at runtime, and handle VA-API/kvazaar compat.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Read a preset env var; the default stays a literal (no per-call leak), only
/// a set override is leaked once.
fn preset_env(
    cache: &'static OnceLock<&'static str>,
    var: &str,
    default: &'static str,
) -> &'static str {
    *cache.get_or_init(|| {
        std::env::var(var)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| -> &'static str { Box::leak(s.into_boxed_str()) })
            .unwrap_or(default)
    })
}

pub(super) fn x264_preset() -> &'static str {
    static CACHE: OnceLock<&'static str> = OnceLock::new();
    preset_env(&CACHE, "SENMEI_X264_PRESET", "veryfast")
}

pub(super) fn kvazaar_preset() -> &'static str {
    static CACHE: OnceLock<&'static str> = OnceLock::new();
    preset_env(&CACHE, "SENMEI_KVAZAAR_PRESET", "veryfast")
}

pub(super) fn x265_preset() -> &'static str {
    static CACHE: OnceLock<&'static str> = OnceLock::new();
    preset_env(&CACHE, "SENMEI_X265_PRESET", "veryfast")
}

/// Hardware encoders to try, HEVC before H.264, per platform.
#[cfg(target_os = "linux")]
pub(super) const HW_ENCODERS: [&str; 8] = [
    "hevc_vaapi",
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "h264_vaapi",
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
];
#[cfg(target_os = "macos")]
pub(super) const HW_ENCODERS: [&str; 2] = ["hevc_videotoolbox", "h264_videotoolbox"];
#[cfg(target_os = "windows")]
pub(super) const HW_ENCODERS: [&str; 6] = [
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(super) const HW_ENCODERS: [&str; 0] = [];

/// Encode on the integrated GPU (iGPU) instead of the discrete GPU.
static PREFER_IGPU: AtomicBool = AtomicBool::new(false);

pub(super) fn set_vaapi_prefer_igpu(v: bool) {
    PREFER_IGPU.store(v, Ordering::Relaxed);
}

/// VA-API device of the discrete GPU by default, or the iGPU when offloading.
pub(super) fn vaapi_device() -> Option<PathBuf> {
    if let Ok(dev) = std::env::var("SENMEI_VAAPI_DEVICE") {
        if !dev.is_empty() {
            let p = Path::new(&dev);
            if p.is_file() {
                return Some(p.to_path_buf());
            }
        }
    }
    let vram = |card: &Path| -> u64 {
        std::fs::read_to_string(card.join("device/mem_info_vram_total"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    let mut cards: Vec<(u32, u64)> = (0..8u32)
        .map(|n| {
            (
                n,
                vram(&Path::new("/sys/class/drm").join(format!("card{n}"))),
            )
        })
        .filter(|(_, v)| *v > 0)
        .collect();
    cards.sort_by(|a, b| {
        if PREFER_IGPU.load(Ordering::Relaxed) {
            a.1.cmp(&b.1)
        } else {
            b.1.cmp(&a.1)
        }
    });
    let dir = Path::new("/dev/dri");
    for (n, _) in cards {
        let render = dir.join(format!("renderD{}", 128 + n));
        if render.is_file() {
            return Some(render);
        }
        let card = dir.join(format!("card{n}"));
        if card.is_file() {
            return Some(card);
        }
    }
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("renderD") {
            return Some(entry.path());
        }
    }
    let card = dir.join("card0");
    card.is_file().then_some(card)
}

/// One-frame test encode at `w × h`; an encoder only counts as available when
/// it actually produces output.
pub(super) fn test_encode(ffmpeg: &Path, codec: &str, w: u32, h: u32) -> bool {
    let mut cmd = crate::process::hidden(ffmpeg);
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    if codec.ends_with("_vaapi") {
        let Some(dev) = vaapi_device() else {
            return false;
        };
        cmd.args(["-init_hw_device", &format!("vaapi=va:{}", dev.display())]);
        cmd.args(["-filter_hw_device", "va"]);
    }
    cmd.args([
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=duration=0.1:size={w}x{h}:rate=10"),
    ]);
    if codec.ends_with("_vaapi") {
        cmd.args(["-vf", "format=nv12,hwupload"]);
    }
    cmd.args(["-c:v", codec, "-f", "null", "-"]);
    match cmd.output() {
        Ok(o) => {
            if !o.status.success() {
                log::warn!(
                    "probe {codec}@{w}x{h} failed: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
            }
            o.status.success()
        }
        Err(e) => {
            log::warn!("probe {codec}@{w}x{h} could not run: {e}");
            false
        }
    }
}

/// Cached per-process verifier (each codec is test-encoded once).
pub(super) fn hw_verifier(ffmpeg: &Path) -> impl Fn(&str) -> bool + '_ {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    move |codec: &str| {
        if let Some(ok) = cache.lock().unwrap().get(codec) {
            return *ok;
        }
        let ok = test_encode(ffmpeg, codec, 640, 480);
        cache.lock().unwrap().insert(codec.to_string(), ok);
        ok
    }
}

/// Drop `flag <value>` pairs listed in `drop`; `rename` maps a flag (keeping
/// its value) before copying.
fn filter_args(args: &[String], drop: &[&str], rename: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        if drop.contains(&args[i].as_str()) {
            i += 2;
        } else if let Some((_, to)) = rename.iter().find(|(f, _)| args[i] == *f) {
            if let Some(v) = args.get(i + 1) {
                out.push(to.to_string());
                out.push(v.clone());
                i += 2;
            } else {
                i += 1;
            }
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    out
}

pub(super) fn kvazaar_compat_args(args: &[String]) -> Vec<String> {
    filter_args(args, &["-tune"], &[])
}

pub(super) fn vaapi_compat_args(args: &[String]) -> Vec<String> {
    filter_args(args, &["-preset", "-tune", "-pix_fmt"], &[("-crf", "-qp")])
}

/// Encoder backend preference, from the frontend's `-senmei_encoder` sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum EncoderPref {
    #[default]
    Auto,
    Hardware,
    Software,
}

/// Resolution-based bitrate for ABR-only encoders (libopenh264).
fn bitrate_kbps(width: u32, height: u32) -> String {
    format!("{}k", width as u64 * height as u64 / 144)
}

/// HW first (verified), then sw chain. HEVC preferred over H.264.
pub(super) fn pick_from_caps(
    caps: &[String],
    width: u32,
    height: u32,
    pref: EncoderPref,
    verify: &dyn Fn(&str) -> bool,
    verify_full: &dyn Fn(&str) -> bool,
) -> (String, Vec<String>) {
    if pref != EncoderPref::Software {
        for codec in HW_ENCODERS {
            if caps.iter().any(|e| e == codec) && verify(codec) && verify_full(codec) {
                return (codec.into(), Vec::new());
            }
        }
    }
    let openh264_ok = width <= 4096 && height <= 4096;
    let chain: &[&str] = if pref == EncoderPref::Software {
        &["libkvazaar", "libx265", "libopenh264", "libx264", "h264"]
    } else {
        &[
            "libkvazaar",
            "libx265",
            "libopenh264",
            "libx264",
            "h264_nvenc",
            "h264",
        ]
    };
    for &codec in chain {
        if codec == "libopenh264" && !openh264_ok {
            continue;
        }
        if caps.iter().any(|e| e == codec) {
            return match codec {
                "libkvazaar" => (
                    codec.into(),
                    vec!["-preset".into(), kvazaar_preset().into()],
                ),
                "libx265" => (codec.into(), vec!["-preset".into(), x265_preset().into()]),
                "libopenh264" => (
                    codec.into(),
                    vec![
                        "-b:v".into(),
                        bitrate_kbps(width, height),
                    ],
                ),
                "libx264" => (codec.into(), vec!["-preset".into(), x264_preset().into()]),
                other => (other.into(), vec![]),
            };
        }
    }
    ("h264".into(), vec![])
}

pub(super) fn override_codec_args(
    codec: &str,
    extra_args: &[String],
    width: u32,
    height: u32,
) -> Vec<String> {
    if codec == "libopenh264" && !extra_args.iter().any(|a| a == "-b:v") {
        vec![
            "-b:v".into(),
            bitrate_kbps(width, height),
        ]
    } else {
        Vec::new()
    }
}

/// Trim the source audio to `[start_ms, start_ms + duration_ms)` into a
/// temp `.m4a` (re-encoded AAC, 0-based PTS).
pub(super) fn extract_audio_range(
    ffmpeg: &Path,
    input: &Path,
    start_ms: u64,
    duration_ms: Option<u64>,
) -> Option<PathBuf> {
    use std::process::Stdio;
    use std::time::{SystemTime, UNIX_EPOCH};
    let tmp = std::env::temp_dir().join(format!(
        "senmei_audio_{}_{}.m4a",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos()
    ));
    let mut cmd = crate::process::hidden(ffmpeg);
    cmd.arg("-y").arg("-loglevel").arg("error");
    if start_ms > 0 {
        cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
    }
    if let Some(dur) = duration_ms {
        cmd.args(["-t", &format!("{:.3}", dur as f64 / 1000.0)]);
    }
    cmd.arg("-i")
        .arg(input)
        .args(["-map", "0:a:0?", "-c:a", "aac"])
        .arg(&tmp)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let ok = cmd.status().map(|s| s.success()).unwrap_or(false);
    let empty = std::fs::metadata(&tmp)
        .map(|m| m.len() == 0)
        .unwrap_or(true);
    if !ok || empty {
        let _ = std::fs::remove_file(&tmp);
        return None;
    }
    Some(tmp)
}
