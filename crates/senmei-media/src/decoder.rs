use std::io::{BufReader, Read};
use std::path::Path;
use std::process::Stdio;

use crate::frame::Frame;
use crate::{Error, Result};

/// HDR→SDR tonemapping policy for the decode stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Tonemap {
    /// Tone-map only when the source is detected as HDR.
    #[default]
    Auto,
    /// Always apply the HDR→SDR filter.
    Always,
    /// Never tone-map.
    Off,
}

/// FFmpeg HDR→SDR filter (zscale + tonemap; LGPL-safe, needs libzimg).
const TONEMAP_VF: &str = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=rgb24";

pub struct Decoder {
    child: std::process::Child,
    stdout: BufReader<std::process::ChildStdout>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub total_frames: u64,
    frame_size: usize,
    remaining: Option<u64>,
}

impl Decoder {
    /// Decode a time range. `start_ms` seeks the input (fast `-ss` before `-i`);
    /// `end_ms` caps the frame count (None = to the end).
    pub fn open_with_range(
        ffmpeg: &Path,
        path: &Path,
        start_ms: u64,
        end_ms: Option<u64>,
        tonemap: Tonemap,
        max_dim: Option<u32>,
    ) -> Result<Self> {
        let info = crate::probe::probe(&crate::ffprobe_next_to(ffmpeg), path)?;
        let fps = info.fps;

        let mut cmd = crate::process::hidden(ffmpeg);
        if start_ms > 0 {
            cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
        }
        // ffmpeg autorotates by default, silently changing the output size;
        // disable it and apply the rotation explicitly.
        if info.rotation != 0 {
            cmd.arg("-noautorotate");
        }
        cmd.arg("-i").arg(path);

        // Tonemap HDR→SDR before the output conversion; rotation last so the
        // decoded frames always match `probe`'s display dimensions.
        let mut filters: Vec<String> = Vec::new();
        if tonemap == Tonemap::Always || (tonemap == Tonemap::Auto && info.is_hdr()) {
            filters.push(TONEMAP_VF.to_owned());
        }
        if info.rotation != 0 {
            let vf = match info.rotation {
                90 => "transpose=2", // 90° counterclockwise
                180 => "hflip,vflip",
                270 => "transpose=1", // 270° cw = 90° clockwise
                other => {
                    return Err(Error::Command(format!(
                        "unsupported rotation: {other} (expected 0/90/180/270)"
                    )));
                }
            };
            filters.push(vf.to_owned());
        }
        // Preview decode budget: downscale only (never upscale) so preview
        // frames match the display instead of the full source resolution.
        let mut out_w = info.width;
        let mut out_h = info.height;
        if let Some(m) = max_dim.filter(|m| *m > 0) {
            let longest = info.width.max(info.height);
            if longest > m {
                let s = m as f64 / longest as f64;
                out_w = ((info.width as f64 * s).round() as u32).max(2) & !1;
                out_h = ((info.height as f64 * s).round() as u32).max(2) & !1;
                filters.push(format!("scale={out_w}:{out_h}"));
            }
        }
        // `-vf` before the output URL: placed after `-`, this ffmpeg build
        // silently drops the graph → misaligned reads → "stripes" in preview.
        if !filters.is_empty() {
            cmd.arg("-vf").arg(filters.join(","));
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);

        // stdin null, or an orphaned ffmpeg would hold the pty open after kill.
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdout".into()))?;

        // Cap on the video-stream duration (copied audio over-reports container).
        let source_dur = if info.video_duration > 0.0 {
            info.video_duration
        } else {
            info.duration
        };
        let dur_ms = (source_dur * 1000.0).round().max(1.0) as u64;
        let remaining = end_ms.map(|end| {
            let end = end.min(dur_ms);
            if end > start_ms {
                (((end - start_ms) as f64 / 1000.0) * fps).round() as u64
            } else {
                0
            }
        });
        let total_frames = remaining.unwrap_or((source_dur * fps).round().max(1.0) as u64);

        Ok(Self {
            child,
            stdout: BufReader::new(stdout),
            width: out_w,
            height: out_h,
            fps,
            total_frames,
            frame_size: (out_w * out_h * 3) as usize,
            remaining,
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        if let Some(r) = self.remaining.as_mut() {
            if *r == 0 {
                return Ok(None);
            }
        }
        let mut buf = vec![0u8; self.frame_size];
        let frame = match self.stdout.read_exact(&mut buf) {
            Ok(()) => Some(Frame {
                width: self.width,
                height: self.height,
                data: buf,
            }),
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => None,
            Err(err) => return Err(err.into()),
        };
        if let Some(r) = self.remaining.as_mut() {
            if frame.is_some() {
                *r = r.saturating_sub(1);
            }
        }
        Ok(frame)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: the `-vf` scale filter must actually apply. When it was
    /// placed after the output URL, this ffmpeg build silently dropped the
    /// graph and emitted unscaled frames; the decoder then read a misaligned
    /// chunk of a larger frame — row-shifted "stripes" in the preview. Only
    /// triggered by sources larger than the preview budget.
    #[test]
    fn max_dim_downscales_matching_direct_ffmpeg() {
        let dir = std::env::temp_dir().join("senmei_decoder_scale_test");
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("big.mp4");
        if !video.exists() {
            let ok = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=1920x1080:r=24:d=1",
                    "-vf",
                    "geq=lum='mod(Y,256)':cb=128:cr=128",
                    "-c:v",
                    "mpeg4",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(&video)
                .status()
                .is_ok_and(|s| s.success());
            if !ok {
                return; // no ffmpeg — skip
            }
        }
        // Direct reference: same seek + scale, one frame.
        let out = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-ss", "0.5", "-i"])
            .arg(&video)
            .args([
                "-vf",
                "scale=1280:720",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .output()
            .unwrap();
        let mut dec = Decoder::open_with_range(
            Path::new("ffmpeg"),
            &video,
            500,
            None,
            Tonemap::Auto,
            Some(1280),
        )
        .expect("open scaled decoder");
        assert_eq!((dec.width, dec.height), (1280, 720));
        let frame = dec.next_frame().expect("frame").expect("some frame");
        assert_eq!(frame.data.len(), 1280 * 720 * 3);
        assert_eq!(
            frame.data, out.stdout,
            "scaled decode must match direct ffmpeg"
        );
    }
}
