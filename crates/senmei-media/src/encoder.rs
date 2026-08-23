use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::{Mutex, OnceLock};

use crate::frame::Frame;
use crate::{Error, Result};

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Option<ChildStderr>,
    /// Owned trimmed-audio temp file (removed on drop); `None` when muxing the
    /// source audio directly or audio is dropped (`-an`).
    temp_audio: Option<PathBuf>,
}

/// x264 speed/quality trade-off. Default `veryfast` keeps 2160p encode ahead of
/// the GPU pipeline; override via `SENMEI_X264_PRESET`.
fn x264_preset() -> &'static str {
    std::env::var("SENMEI_X264_PRESET")
        .unwrap_or_else(|_| "veryfast".into())
        .leak()
}

/// kvazaar (HEVC) speed/quality trade-off; override via `SENMEI_KVAZAAR_PRESET`.
fn kvazaar_preset() -> &'static str {
    std::env::var("SENMEI_KVAZAAR_PRESET")
        .unwrap_or_else(|_| "veryfast".into())
        .leak()
}

/// Hardware encoders to try, HEVC before H.264, per platform. Only used when a
/// runtime test encode confirms the encoder actually works (they are listed in
/// `-encoders` even without a GPU and then fail at runtime).
#[cfg(target_os = "linux")]
const HW_ENCODERS: [&str; 8] = [
    "hevc_vaapi", "hevc_nvenc", "hevc_qsv", "hevc_amf",
    "h264_vaapi", "h264_nvenc", "h264_qsv", "h264_amf",
];
#[cfg(target_os = "macos")]
const HW_ENCODERS: [&str; 2] = ["hevc_videotoolbox", "h264_videotoolbox"];
#[cfg(target_os = "windows")]
const HW_ENCODERS: [&str; 6] = [
    "hevc_nvenc", "hevc_qsv", "hevc_amf", "h264_nvenc", "h264_qsv", "h264_amf",
];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const HW_ENCODERS: [&str; 0] = [];

/// First VA-API render node (preferred) or card, if any.
fn vaapi_device() -> Option<std::path::PathBuf> {
    let dir = Path::new("/dev/dri");
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("renderD") {
            return Some(entry.path());
        }
    }
    let card = dir.join("card0");
    card.is_file().then_some(card)
}

/// One-frame test encode; an encoder only counts as available when it actually
/// produces output (VA-API gets an explicit device + hwupload).
fn test_encode(ffmpeg: &Path, codec: &str) -> bool {
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    if codec.ends_with("_vaapi") {
        let Some(dev) = vaapi_device() else { return false };
        cmd.arg(format!("-init_hw_device vaapi=va:{}", dev.display()));
        cmd.args(["-filter_hw_device", "va"]);
    }
    cmd.args(["-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x48:rate=10"]);
    if codec.ends_with("_vaapi") {
        cmd.args(["-vf", "format=nv12,hwupload"]);
    }
    cmd.args(["-c:v", codec, "-f", "null", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

/// Cached per-process verifier (each codec is test-encoded once).
fn hw_verifier(ffmpeg: &Path) -> impl Fn(&str) -> bool + '_ {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    move |codec: &str| {
        if let Some(ok) = cache.lock().unwrap().get(codec) {
            return *ok;
        }
        let ok = test_encode(ffmpeg, codec);
        cache.lock().unwrap().insert(codec.to_string(), ok);
        ok
    }
}

/// Pick the best video encoder available in `ffmpeg`. Verified hardware
/// encoders come first (fast; HEVC before H.264); otherwise the LGPL-safe
/// software chain: libkvazaar (HEVC, BSD — ships in the bundled LGPL builds),
/// then libopenh264 (H.264), then libx264 (GPL-only, works on GPU-less
/// runners), then the native fallback. kvazaar/x264 default to quality-based
/// rate control; libopenh264 is fixed-bitrate ABR, so it gets a
/// resolution-based `-b:v` (~14 Mbps @1080p, 144 bits/px) — the caller's
/// `extra_args` are appended later and can override it.
fn pick_from_caps(
    caps: &[String],
    width: u32,
    height: u32,
    verify: &dyn Fn(&str) -> bool,
) -> (String, Vec<String>) {
    for codec in HW_ENCODERS {
        if caps.iter().any(|e| e == codec) && verify(codec) {
            return (codec.into(), Vec::new());
        }
    }
    // libopenh264 hard-caps at 4096x4096 — for larger frames skip it so the
    // chain falls through to libx264 (or the native h264 fallback) instead of
    // failing the encode at >4K output (e.g. x4 from 1080p).
    let openh264_ok = width <= 4096 && height <= 4096;
    for codec in ["libkvazaar", "libopenh264", "libx264", "h264_nvenc", "h264"] {
        if codec == "libopenh264" && !openh264_ok {
            continue;
        }
        if caps.iter().any(|e| e == codec) {
            return match codec {
                "libkvazaar" => (
                    codec.into(),
                    vec!["-preset".into(), kvazaar_preset().into()],
                ),
                "libopenh264" => (
                    codec.into(),
                    vec![
                        "-b:v".into(),
                        format!("{}k", width as u64 * height as u64 / 144),
                    ],
                ),
                "libx264" => (codec.into(), vec!["-preset".into(), x264_preset().into()]),
                other => (other.into(), vec![]),
            };
        }
    }
    ("h264".into(), vec![])
}

/// Trim the source audio to `[start_ms, start_ms + duration_ms)` into a
/// temp `.m4a` (re-encoded AAC, 0-based PTS) so the encoder can stream-copy it
/// in sync with the ranged video. `None` when the source has no audio or the
/// extraction fails — the caller falls back to muxing the source directly.
fn extract_audio_range(
    ffmpeg: &Path,
    input: &Path,
    start_ms: u64,
    duration_ms: Option<u64>,
) -> Option<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let tmp = std::env::temp_dir().join(format!(
        "senmei_audio_{}_{}.m4a",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_nanos()
    ));
    let mut cmd = Command::new(ffmpeg);
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
    let empty = std::fs::metadata(&tmp).map(|m| m.len() == 0).unwrap_or(true);
    if !ok || empty {
        let _ = std::fs::remove_file(&tmp);
        return None;
    }
    Some(tmp)
}

/// Default extra args when the caller overrides `-c:v` (the frontend codec
/// dropdown). libopenh264 is bitrate-only (ABR), so it gets a resolution-based
/// `-b:v` (same formula as `pick_video_encoder`) unless already provided.
fn override_codec_args(codec: &str, extra_args: &[String], width: u32, height: u32) -> Vec<String> {
    if codec == "libopenh264" && !extra_args.iter().any(|a| a == "-b:v") {
        vec![
            "-b:v".into(),
            format!("{}k", width as u64 * height as u64 / 144),
        ]
    } else {
        Vec::new()
    }
}

impl Encoder {
    /// `extra_args` are appended after the defaults (before the output path), so
    /// user-supplied codec/filter options override the built-in defaults.
    /// `input` is a second ffmpeg input whose audio is mapped (`-map 1:a:0?`,
    /// optional) so the output keeps the source sound unless `-an` is passed.
    /// `start_ms` seeks the audio input so it stays in sync with a ranged render;
    /// `duration_ms` bounds it (`-t`) to the same range — without it the copied
    /// audio input runs to the end of the source and ffmpeg never exits after
    /// the (shorter) video pipe ends.
    pub fn open(
        ffmpeg: &Path,
        input: &Path,
        path: &Path,
        width: u32,
        height: u32,
        fps: f64,
        start_ms: u64,
        duration_ms: Option<u64>,
        extra_args: &[String],
    ) -> Result<Self> {
        let caps = crate::ffmpeg::probe(ffmpeg).encoders;
        let verify = hw_verifier(ffmpeg);
        let (mut video_codec, mut codec_args) = pick_from_caps(&caps, width, height, &verify);
        // Strip any caller-supplied `-c:v` from extra_args: we always pass the
        // codec ourselves (below) so it can be validated against the available
        // encoders (the frontend maps H.265→libkvazaar even on builds without
        // it) and so ffmpeg doesn't see two `-c:v` options.
        let mut extra_args = extra_args.to_vec();
        if let Some(pos) = extra_args.windows(2).position(|w| w[0] == "-c:v") {
            let codec = extra_args[pos + 1].clone();
            extra_args.drain(pos..pos + 2);
            if caps.iter().any(|e| *e == codec) {
                video_codec = codec.clone();
                codec_args = override_codec_args(&codec, &extra_args, width, height);
            } else {
                log::warn!("encoder `{codec}` unavailable; falling back to `{video_codec}`");
            }
        }
        // VA-API needs an explicit device + hardware upload; NVENC/QSV/AMF/VT
        // take ordinary frames and handle the upload themselves.
        let vaapi = video_codec.ends_with("_vaapi").then(vaapi_device).flatten();
        let mut cmd = Command::new(ffmpeg);
        cmd.arg("-y");
        if let Some(dev) = &vaapi {
            cmd.arg(format!("-init_hw_device vaapi=va:{}", dev.display()));
            cmd.args(["-filter_hw_device", "va"]);
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"]);
        // Muxing the source audio directly with `-ss`/`-t` between the two
        // inputs + `-copyts` is unreliable: the seeked audio keeps its source
        // PTS (dropped/desynced by `-shortest`), and some containers ignore the
        // seek entirely (audio from the start of the file). Extract the exact
        // range to a temp file first (re-encoded, 0-based), then stream-copy it
        // in — deterministic regardless of the source container.
        let want_audio = !extra_args.iter().any(|a| a == "-an");
        let mut temp_audio: Option<PathBuf> = None;
        if want_audio && (start_ms > 0 || duration_ms.is_some()) {
            temp_audio = extract_audio_range(ffmpeg, input, start_ms, duration_ms);
        }
        if let Some(tmp) = &temp_audio {
            cmd.arg("-i").arg(tmp);
        } else {
            if start_ms > 0 {
                cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
            }
            if let Some(dur) = duration_ms {
                cmd.args(["-t", &format!("{:.3}", dur as f64 / 1000.0)]);
            }
            cmd.arg("-i").arg(input);
        }
        cmd
            // Keep the pipe video's 0-based PTS: without this the muxer re-bases
            // the output to the seeked-and-copied audio and shifts the video by
            // the audio's start offset (e.g. 0.67 s), breaking the monitor's
            // `source - inMs` frame mapping in compare/result.
            .arg("-copyts")
            .args(["-map", "0:v:0", "-map", "1:a:0?"])
            // Stop at the shortest stream: without this the copied source audio
            // runs past a ranged render and the container reports the audio's
            // (much longer) duration, breaking seeks near the video end.
            .args(["-shortest"])
            .args(if temp_audio.is_some() {
                // The temp file is already the exact, 0-based range.
                vec!["-c:a".to_string(), "copy".to_string()]
            } else {
                Vec::new()
            })
            .args(["-c:v", &video_codec])
            .args(codec_args)
            .args(if vaapi.is_some() {
                ["-vf".to_string(), "format=nv12,hwupload".to_string()]
            } else {
                ["-pix_fmt".to_string(), "yuv420p".to_string()]
            })
            .args(&extra_args)
            .arg(path)
            // stdout null: the encoder writes to the output file, not the
            // terminal — inheriting stdout would leave the pty held by an
            // orphaned ffmpeg after the app is killed.
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdin".into()))?;
        let stderr = child.stderr.take();

        Ok(Self {
            child,
            stdin: Some(stdin),
            stderr,
            temp_audio,
        })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if let Some(stdin) = self.stdin.as_mut() {
            if let Err(e) = stdin.write_all(&frame.data) {
                // The child closed the pipe (exited) — reap it first so the
                // stderr read below hits EOF instead of blocking, then report
                // the real reason instead of a bare "Broken pipe".
                let _ = self.child.kill();
                let _ = self.child.wait();
                let stderr = self.read_stderr();
                return Err(Error::Command(if stderr.is_empty() {
                    format!("ffmpeg encode write failed: {e}")
                } else {
                    format!("ffmpeg encode write failed: {e}\n{stderr}")
                }));
            }
        }
        Ok(())
    }

    /// Drain the child's stderr (already buffered once it has exited). ffmpeg
    /// prints its config banner first, so keep only the tail (the real error).
    fn read_stderr(&mut self) -> String {
        use std::io::Read;
        let mut out = String::new();
        if let Some(mut e) = self.stderr.take() {
            let _ = e.read_to_string(&mut out);
        }
        const TAIL: usize = 12;
        let lines: Vec<&str> = out.lines().collect();
        let tail = if lines.len() > TAIL {
            &lines[lines.len() - TAIL..]
        } else {
            &lines[..]
        };
        tail.join("\n").trim().to_string()
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            let stderr = self.read_stderr();
            Err(Error::Command(if stderr.is_empty() {
                format!("ffmpeg encode exited with {status}")
            } else {
                format!("ffmpeg encode exited with {status}:\n{stderr}")
            }))
        }
    }

    /// Abort the encoder immediately (cancel path): kill ffmpeg and reap it so
    /// the pipeline frees its resources without waiting for a normal mux
    /// finalize. The caller discards the output file.
    pub fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(tmp) = self.temp_audio.take() {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_codec_sets_bitrate_for_openh264_only() {
        // libopenh264 is ABR-only: the override adds a resolution-based `-b:v`
        // unless the caller already passed one; other codecs get no defaults.
        let w = 1920u32;
        let h = 1080u32;
        let base = ["-c:v".into(), "libopenh264".into()];
        assert_eq!(
            override_codec_args("libopenh264", &base, w, h),
            vec!["-b:v".to_string(), "14400k".to_string()]
        );
        let with_bv = [
            "-c:v".into(),
            "libopenh264".into(),
            "-b:v".into(),
            "1000k".into(),
        ];
        assert_eq!(
            override_codec_args("libopenh264", &with_bv, w, h),
            Vec::<String>::new()
        );
        assert_eq!(
            override_codec_args("libkvazaar", &base, w, h),
            Vec::<String>::new()
        );
        assert_eq!(
            override_codec_args("libsvtav1", &base, w, h),
            Vec::<String>::new()
        );
    }

    #[test]
    fn verified_hw_encoder_beats_software() {
        if HW_ENCODERS.is_empty() {
            return;
        }
        let mut caps = vec!["libkvazaar".to_string()];
        caps.extend(HW_ENCODERS.iter().map(|c| c.to_string()));
        let (codec, _) = pick_from_caps(&caps, 1920, 1080, &|c| c == HW_ENCODERS[0]);
        assert_eq!(codec, HW_ENCODERS[0]);
    }

    #[test]
    fn listed_but_unverified_hw_falls_back() {
        let mut caps = vec!["libkvazaar".to_string()];
        caps.extend(HW_ENCODERS.iter().map(|c| c.to_string()));
        let (codec, args) = pick_from_caps(&caps, 1920, 1080, &|_| false);
        assert_eq!(codec, "libkvazaar");
        assert!(args.contains(&"-preset".to_string()));
    }

    #[test]
    fn hevc_hw_comes_before_h264_hw() {
        if HW_ENCODERS.is_empty() {
            return;
        }
        assert!(
            HW_ENCODERS[0].starts_with("hevc_"),
            "HEVC first in {HW_ENCODERS:?}"
        );
        let caps: Vec<String> = HW_ENCODERS.iter().map(|c| c.to_string()).collect();
        let (codec, _) = pick_from_caps(&caps, 1920, 1080, &|_| true);
        assert_eq!(codec, HW_ENCODERS[0]);
    }

    /// End-to-end encode through the selected (LGPL-safe) codec. Skipped unless
    /// `SENMEI_FFMPEG` points at a real ffmpeg (e.g. the pinned BtbN LGPL build).
    #[test]
    fn encodes_through_selected_codec() {
        let Some(ff) = std::env::var("SENMEI_FFMPEG")
            .ok()
            .filter(|p| !p.is_empty())
        else {
            eprintln!("SENMEI_FFMPEG not set, skipping");
            return;
        };
        let ff = Path::new(&ff);
        let (codec, _args) = pick_from_caps(&crate::ffmpeg::probe(ff).encoders, 64, 64, &|_| false);
        assert!(
            ["libkvazaar", "libopenh264", "libx264", "h264"].contains(&codec.as_str()),
            "unexpected codec {codec}"
        );

        let dir = std::env::temp_dir().join("senmei-enc-test");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.mp4");
        let out = dir.join("out.mp4");
        let _ = std::fs::remove_file(&out);
        // Valid input (2 s silent AAC) so the optional `-map 1:a:0?` + `-shortest`
        // don't kill the pipe: video (30 frames @30fps = 1 s) is the shortest.
        let make = Command::new(ff)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                "2",
                "-c:a",
                "aac",
                "-ar",
                "44100",
                "-ac",
                "1",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(make.success(), "failed to create test input");
        let mut enc = Encoder::open(&ff, &input, &out, 64, 64, 30.0, 0, None, &[]).unwrap();
        let frame = Frame {
            width: 64,
            height: 64,
            data: vec![0u8; 64 * 64 * 3],
        };
        for _ in 0..30 {
            enc.write_frame(&frame).unwrap();
        }
        enc.finish().unwrap();
        assert!(out.exists() && out.metadata().unwrap().len() > 0);
        let status = Command::new(ff)
            .args(["-v", "error", "-i"])
            .arg(&out)
            .args(["-f", "null", "-"])
            .status()
            .unwrap();
        assert!(status.success(), "encoded output not decodable");
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&input);
    }
}
