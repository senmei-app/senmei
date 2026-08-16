use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::frame::Frame;
use crate::{Error, Result};

pub struct Decoder {
    child: std::process::Child,
    stdout: BufReader<std::process::ChildStdout>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub total_frames: u64,
    frame_size: usize,
}

impl Decoder {
    pub fn open(ffmpeg: &Path, path: &Path) -> Result<Self> {
        let info = crate::probe::probe(path)?;

        let mut child = Command::new(ffmpeg)
            .arg("-i")
            .arg(path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdout".into()))?;

        Ok(Self {
            child,
            stdout: BufReader::new(stdout),
            width: info.width,
            height: info.height,
            fps: info.fps,
            total_frames: (info.duration * info.fps).round() as u64,
            frame_size: (info.width * info.height * 3) as usize,
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let mut buf = vec![0u8; self.frame_size];
        match self.stdout.read_exact(&mut buf) {
            Ok(()) => Ok(Some(Frame {
                width: self.width,
                height: self.height,
                data: buf,
            })),
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
