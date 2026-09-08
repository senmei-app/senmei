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
        // Auto deinterlace: apply yadif when the source is interlaced.
        if info.is_interlaced() {
            filters.push("yadif=0:-1:0".to_owned());
        }
        // Auto desqueeze: when PAR != 1:1, scale to square pixels at decode
        // time. Applied before rotation so PAR correction uses storage dims.
        let mut out_w = info.width;
        let mut out_h = info.height;
        if let Some((tw, th)) = auto_desqueeze_target(&info) {
            filters.push(format!("scale={tw}:{th}:flags=bilinear"));
            out_w = tw;
            out_h = th;
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
        if let Some(m) = max_dim.filter(|m| *m > 0) {
            let longest = out_w.max(out_h);
            if longest > m {
                let s = m as f64 / longest as f64;
                out_w = ((out_w as f64 * s).round() as u32).max(2) & !1;
                out_h = ((out_h as f64 * s).round() as u32).max(2) & !1;
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

/// Compute the desqueeze target dimensions from the source PAR.
/// Returns `Some((target_w, target_h))` when the source has non-square pixels
/// and the target width is a reasonable anamorphic size (even, ≤2048).
fn auto_desqueeze_target(info: &crate::probe::VideoInfo) -> Option<(u32, u32)> {
    let par = info.par.as_deref()?;
    let (pn, pm) = parse_ratio(par)?;
    if pm == 0 || pn == pm {
        return None; // PAR 1:1 or invalid
    }
    let target_w = ((pn as f64 * info.width as f64 / pm as f64).round() as u32 + 1) & !1;
    if target_w <= info.width || target_w > 2048 {
        return None; // not anamorphic or unreasonably large
    }
    Some((target_w, info.height))
}

/// Parse a ratio string like "64:45" into `(num, den)`.
fn parse_ratio(s: &str) -> Option<(u32, u32)> {
    let (n, d) = s.split_once(':')?;
    let n: u32 = n.trim().parse().ok()?;
    let d: u32 = d.trim().parse().ok()?;
    if d == 0 {
        None
    } else {
        Some((n, d))
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

    #[test]
    fn auto_desqueeze_pal_16by9() {
        use crate::probe::VideoInfo;
        let info = VideoInfo {
            width: 720,
            height: 576,
            fps: 25.0,
            duration: 1.0,
            video_duration: 1.0,
            rotation: 0,
            color_transfer: None,
            color_primaries: None,
            video_codec: None,
            audio_codec: None,
            pix_fmt: None,
            par: Some("64:45".into()),
            dar: Some("16:9".into()),
            field_order: None,
            audio_tracks: vec![],
            subtitle_tracks: vec![],
        };
        assert_eq!(auto_desqueeze_target(&info), Some((1024, 576)));
    }

    #[test]
    fn auto_desqueeze_pal_4by3() {
        use crate::probe::VideoInfo;
        let info = VideoInfo {
            width: 720,
            height: 576,
            fps: 25.0,
            duration: 1.0,
            video_duration: 1.0,
            rotation: 0,
            color_transfer: None,
            color_primaries: None,
            video_codec: None,
            audio_codec: None,
            pix_fmt: None,
            par: Some("16:15".into()),
            dar: Some("4:3".into()),
            field_order: None,
            audio_tracks: vec![],
            subtitle_tracks: vec![],
        };
        assert_eq!(auto_desqueeze_target(&info), Some((768, 576)));
    }

    #[test]
    fn auto_desqueeze_skips_square_pixels() {
        use crate::probe::VideoInfo;
        let info = VideoInfo {
            width: 1920,
            height: 1080,
            fps: 24.0,
            duration: 1.0,
            video_duration: 1.0,
            rotation: 0,
            color_transfer: None,
            color_primaries: None,
            video_codec: None,
            audio_codec: None,
            pix_fmt: None,
            par: Some("1:1".into()),
            dar: Some("16:9".into()),
            field_order: None,
            audio_tracks: vec![],
            subtitle_tracks: vec![],
        };
        assert_eq!(auto_desqueeze_target(&info), None);
    }

    #[test]
    fn auto_desqueeze_skips_no_par() {
        use crate::probe::VideoInfo;
        let info = VideoInfo {
            width: 720,
            height: 576,
            fps: 25.0,
            duration: 1.0,
            video_duration: 1.0,
            rotation: 0,
            color_transfer: None,
            color_primaries: None,
            video_codec: None,
            audio_codec: None,
            pix_fmt: None,
            par: None,
            dar: None,
            field_order: None,
            audio_tracks: vec![],
            subtitle_tracks: vec![],
        };
        assert_eq!(auto_desqueeze_target(&info), None);
    }

    #[test]
    fn parse_ratio_valid() {
        assert_eq!(parse_ratio("64:45"), Some((64, 45)));
        assert_eq!(parse_ratio("1:1"), Some((1, 1)));
        assert_eq!(parse_ratio("16:15"), Some((16, 15)));
    }

    #[test]
    fn parse_ratio_invalid() {
        assert_eq!(parse_ratio(""), None);
        assert_eq!(parse_ratio("abc"), None);
        assert_eq!(parse_ratio("1:0"), None);
    }
}
