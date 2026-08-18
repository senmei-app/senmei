use std::io::Write;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::frame::Frame;
use crate::{Error, Result};

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
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

/// Pick the best video encoder available in `ffmpeg`, preferring LGPL-safe
/// codecs: libkvazaar (HEVC, BSD — ships in the bundled LGPL builds; better
/// compression than H.264 at 2160p), then libopenh264 (H.264), then hardware
/// (h264_nvenc), then libx264 (GPL-only — present in system GPL builds), then
/// the native fallback. kvazaar/x264 default to quality-based rate control;
/// libopenh264 is fixed-bitrate ABR, so it gets a resolution-based `-b:v`
/// (~14 Mbps @1080p, 144 bits/px) — the caller's `extra_args` are appended
/// later and can override it.
fn pick_video_encoder(ffmpeg: &Path, width: u32, height: u32) -> (String, Vec<String>) {
    let caps = crate::ffmpeg::probe(ffmpeg).encoders;
    for codec in ["libkvazaar", "libopenh264", "h264_nvenc", "libx264", "h264"] {
        if caps.iter().any(|e| e == codec) {
            return match codec {
                "libkvazaar" => (codec.into(), vec!["-preset".into(), kvazaar_preset().into()]),
                "libopenh264" => (
                    codec.into(),
                    vec!["-b:v".into(), format!("{}k", width as u64 * height as u64 / 144)],
                ),
                "libx264" => (codec.into(), vec!["-preset".into(), x264_preset().into()]),
                other => (other.into(), vec![]),
            };
        }
    }
    ("h264".into(), vec![])
}

/// Default extra args when the caller overrides `-c:v` (the frontend codec
/// dropdown). libopenh264 is bitrate-only (ABR), so it gets a resolution-based
/// `-b:v` (same formula as `pick_video_encoder`) unless already provided.
fn override_codec_args(codec: &str, extra_args: &[String], width: u32, height: u32) -> Vec<String> {
    if codec == "libopenh264" && !extra_args.iter().any(|a| a == "-b:v") {
        vec!["-b:v".into(), format!("{}k", width as u64 * height as u64 / 144)]
    } else {
        Vec::new()
    }
}

impl Encoder {
    /// `extra_args` are appended after the defaults (before the output path), so
    /// user-supplied codec/filter options override the built-in defaults.
    /// `input` is a second ffmpeg input whose audio is mapped (`-map 1:a:0?`,
    /// optional) so the output keeps the source sound unless `-an` is passed.
    /// `start_ms` seeks the audio input so it stays in sync with a ranged render.
    pub fn open(
        ffmpeg: &Path,
        input: &Path,
        path: &Path,
        width: u32,
        height: u32,
        fps: f64,
        start_ms: u64,
        extra_args: &[String],
    ) -> Result<Self> {
        let (video_codec, mut codec_args) = pick_video_encoder(ffmpeg, width, height);
        // A caller-supplied `-c:v` fully owns the codec: drop the default
        // codec's args (e.g. libkvazaar's `-preset`) and apply the override's
        // own defaults, so a GPL/ABR mismatch never leaks through.
        if let Some(codec) = extra_args
            .windows(2)
            .find(|w| w[0] == "-c:v")
            .map(|w| w[1].as_str())
        {
            codec_args = override_codec_args(codec, extra_args, width, height);
        }
        let mut cmd = Command::new(ffmpeg);
        cmd.arg("-y")
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"]);
        if start_ms > 0 {
            cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
        }
        cmd.arg("-i")
            .arg(input)
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
            .args(["-c:v", &video_codec])
            .args(codec_args)
            .args(["-pix_fmt", "yuv420p"])
            .args(extra_args)
            .arg(path)
            .stdin(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = cmd.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdin".into()))?;

        Ok(Self {
            child,
            stdin: Some(stdin),
        })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if let Some(stdin) = self.stdin.as_mut() {
            stdin.write_all(&frame.data)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(Error::Command(format!(
                "ffmpeg encode exited with {status}"
            )))
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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
        let with_bv = ["-c:v".into(), "libopenh264".into(), "-b:v".into(), "1000k".into()];
        assert_eq!(override_codec_args("libopenh264", &with_bv, w, h), Vec::<String>::new());
        assert_eq!(override_codec_args("libkvazaar", &base, w, h), Vec::<String>::new());
        assert_eq!(override_codec_args("libsvtav1", &base, w, h), Vec::<String>::new());
    }

    /// End-to-end encode through the selected (LGPL-safe) codec. Skipped unless
    /// `SENMEI_FFMPEG` points at a real ffmpeg (e.g. the pinned BtbN LGPL build).
    #[test]
    fn encodes_through_selected_codec() {
        let Some(ff) = std::env::var("SENMEI_FFMPEG").ok().filter(|p| !p.is_empty()) else {
            eprintln!("SENMEI_FFMPEG not set, skipping");
            return;
        };
        let ff = Path::new(&ff);
        let (codec, _args) = pick_video_encoder(ff, 64, 64);
        assert!(
            ["libkvazaar", "libopenh264", "h264_nvenc", "libx264", "h264"].contains(&codec.as_str()),
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
            .args(["-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "2", "-c:a", "aac", "-ar", "44100", "-ac", "1"])
            .arg(&input)
            .status()
            .unwrap();
        assert!(make.success(), "failed to create test input");
        let mut enc = Encoder::open(&ff, &input, &out, 64, 64, 30.0, 0, &[]).unwrap();
        let frame = Frame { width: 64, height: 64, data: vec![0u8; 64 * 64 * 3] };
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
