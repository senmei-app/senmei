use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};

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
    ) -> Result<Self> {
        let info = crate::probe::probe(&crate::ffprobe_next_to(ffmpeg), path)?;
        let fps = info.fps;

        let mut cmd = Command::new(ffmpeg);
        if start_ms > 0 {
            cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
        }
        cmd.arg("-i")
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);

        // Tonemap HDR→SDR before the output conversion; rotation last so the
        // decoded frames always match `probe`'s display dimensions.
        let mut filters: Vec<&str> = Vec::new();
        if tonemap == Tonemap::Always || (tonemap == Tonemap::Auto && info.is_hdr()) {
            filters.push(TONEMAP_VF);
        }
        // ffmpeg autorotates by default (DisplayMatrix), which would silently
        // change the output size away from the probed one. Disable that and
        // apply the rotation explicitly. The filter per rotation is verified
        // byte-identical against ffmpeg's own autorotation.
        if info.rotation != 0 {
            cmd.arg("-noautorotate");
            let vf = match info.rotation {
                90 => "transpose=2", // 90° counterclockwise
                180 => "hflip,vflip",
                270 => "transpose=1", // 270° cw = 90° clockwise
                _ => unreachable!(),
            };
            filters.push(vf);
        }
        if !filters.is_empty() {
            cmd.arg("-vf").arg(filters.join(","));
        }

        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdout".into()))?;

        let dur_ms = (info.duration * 1000.0).round().max(1.0) as u64;
        let remaining = end_ms.map(|end| {
            let end = end.min(dur_ms);
            if end > start_ms {
                (((end - start_ms) as f64 / 1000.0) * fps).round() as u64
            } else {
                0
            }
        });
        let total_frames = remaining.unwrap_or((info.duration * fps).round().max(1.0) as u64);

        Ok(Self {
            child,
            stdout: BufReader::new(stdout),
            width: info.width,
            height: info.height,
            fps,
            total_frames,
            frame_size: (info.width * info.height * 3) as usize,
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
