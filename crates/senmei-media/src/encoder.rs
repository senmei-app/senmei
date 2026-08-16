use std::io::Write;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::frame::Frame;
use crate::{Error, Result};

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Encoder {
    pub fn open(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .arg("-y")
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::command_failed("failed to capture ffmpeg stdin".into()))?;

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
            Err(Error::command_failed(format!(
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
