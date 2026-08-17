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

impl Encoder {
    /// `extra_args` are appended after the defaults (before the output path), so
    /// user-supplied codec/filter options override the built-in x264 defaults.
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
            .args(["-map", "0:v:0", "-map", "1:a:0?"])
            .args(["-c:v", "libx264", "-preset", x264_preset(), "-pix_fmt", "yuv420p"])
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
