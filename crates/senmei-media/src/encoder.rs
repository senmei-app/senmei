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
    pub fn open(ffmpeg: &Path, path: &Path, width: u32, height: u32, fps: f64) -> Result<Self> {
        let mut child = Command::new(ffmpeg)
            .arg("-y")
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"])
            .args(["-c:v", "libx264", "-preset", x264_preset(), "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

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
