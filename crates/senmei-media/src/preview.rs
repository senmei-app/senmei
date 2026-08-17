//! Single-frame extraction for the preview monitor.

use std::path::Path;
use std::process::Command;

use crate::{Error, Result};

/// Extract one frame at `pos_secs` as a JPEG (codec-agnostic via ffmpeg).
/// `ffmpeg` must be the resolved binary (same one the pipeline uses).
pub fn extract_frame(ffmpeg: &Path, path: &Path, pos_secs: f64) -> Result<Vec<u8>> {
    let output = Command::new(ffmpeg)
        .args(["-ss", &format!("{pos_secs:.3}"), "-i"])
        .arg(path)
        // `-strict unofficial`: the mjpeg encoder refuses limited-range (tv) YUV
        // (e.g. libx265/hevc output) without it ("Non full-range YUV is non-standard").
        .args(["-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-strict", "unofficial", "-"])
        .output()?;
    if !output.status.success() {
        return Err(Error::Command(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(output.stdout)
}
