//! Single-frame extraction + streaming PCM audio for the preview monitor.

use std::io::Read;

use crate::{Error, Result};

/// Encode an RGB24 frame as PNG bytes (full-range; no ffmpeg round-trip).
pub fn encode_png(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc
            .write_header()
            .map_err(|e| Error::Command(e.to_string()))?;
        w.write_image_data(rgb)
            .map_err(|e| Error::Command(e.to_string()))?;
    }
    Ok(out)
}

/// Streaming PCM — no re-encode, no file, no rodio-codec; caller owns the
/// ffmpeg child for kill-on-seek.
pub struct PcmPipe {
    child: std::process::Child,
}

impl PcmPipe {
    /// Kill ffmpeg; the reader thread exits as soon as stdout closes.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn the PCM pipe: returns the child (for kill) + a chunk channel.
pub fn stream_pcm(
    ffmpeg: &std::path::Path,
    input: &std::path::Path,
    position_ms: f64,
    sample_rate: u32,
) -> Result<(PcmPipe, std::sync::mpsc::Receiver<Vec<u8>>)> {
    let mut child = crate::process::hidden(ffmpeg)
        .args([
            "-ss",
            &format!("{:.3}", position_ms.max(0.0) / 1000.0),
            "-i",
        ])
        .arg(input)
        .args([
            "-vn",
            "-ac",
            "2",
            "-ar",
            &sample_rate.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::Command("no ffmpeg stdout".into()))?;
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut r = std::io::BufReader::new(stdout);
        let mut buf = [0u8; 65536];
        loop {
            match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok((PcmPipe { child }, rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stream hands back PCM chunks for any source codec (here a sine
    /// WAV); stopping the pipe ends it.
    #[test]
    fn stream_pcm_feeds_samples() {
        let ffmpeg = std::env::var("SENMEI_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let dir = std::env::temp_dir();
        let src = dir.join("senmei_stream_src.wav");
        let ok = std::process::Command::new(&ffmpeg)
            .args([
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&src)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "failed to generate source audio");
        let (mut pipe, rx) =
            stream_pcm(std::path::Path::new(&ffmpeg), &src, 0.0, 48_000).expect("spawn stream");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut got = 0usize;
        while got == 0 && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(chunk) => got = chunk.len(),
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
            }
        }
        assert!(got > 0, "no PCM chunks arrived from the pipe");
        pipe.stop();
        let _ = std::fs::remove_file(&src);
    }
}
