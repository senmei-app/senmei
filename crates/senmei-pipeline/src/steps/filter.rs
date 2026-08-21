use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use senmei_media::Frame;

use crate::Step;

/// Filter: run each frame through an FFmpeg filter graph over a rawvideo pipe.
/// Frame-preserving only (1:1) — a filter that changes the output size is
/// rejected. Spawns one short-lived `ffmpeg -i - -vf <filter> -` per frame
/// (stateless, no pipe deadlock); sits wherever it is placed in `Vec<Step>`
/// (pre/post/between other steps).
pub struct Filter {
    filter: String,
    ffmpeg: PathBuf,
}

impl Filter {
    pub fn new(filter: impl Into<String>, ffmpeg: impl Into<PathBuf>) -> Self {
        Self {
            filter: filter.into(),
            ffmpeg: ffmpeg.into(),
        }
    }
}

impl Step for Filter {
    fn name(&self) -> &'static str {
        "filter"
    }

    fn process(&mut self, frame: &mut Frame) -> crate::Result<bool> {
        let (w, h) = (frame.width, frame.height);
        let size = format!("{w}x{h}");
        let mut child = Command::new(&self.ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                size.as_str(),
                "-i",
                "-",
                "-vf",
                self.filter.as_str(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(crate::Error::Io)?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| crate::Error::new("filter: no stdin"))?;
        stdin.write_all(&frame.data)?;
        drop(stdin); // EOF → ffmpeg flushes the filtergraph
        let out = child.wait_with_output().map_err(crate::Error::Io)?;
        if !out.status.success() {
            let msg = String::from_utf8_lossy(&out.stderr);
            return Err(crate::Error::new(format!(
                "ffmpeg filter failed: {}",
                msg.trim()
            )));
        }
        if out.stdout.len() != frame.data.len() {
            return Err(crate::Error::new(format!(
                "filter changed frame size ({} -> {} bytes); only frame-preserving filters are supported",
                frame.data.len(),
                out.stdout.len()
            )));
        }
        frame.data = out.stdout;
        Ok(true)
    }
}
