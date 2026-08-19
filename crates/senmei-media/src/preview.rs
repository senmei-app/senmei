//! Single-frame extraction for the preview monitor.

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

/// Extract the source audio track as MP3 for the fallback preview `<audio>`:
/// webviews can't play the source, but every webview plays MP3 (`audio/mpeg`).
/// WebM/Opus failed with MEDIA_ERR_SRC_NOT_SUPPORTED in WebKitGTK `<audio>`
/// (video/* mime), so stick to a native audio container.
pub fn extract_audio(
    ffmpeg: &std::path::Path,
    input: &std::path::Path,
    out: &std::path::Path,
) -> Result<()> {
    let status = std::process::Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vn", "-c:a", "libmp3lame", "-b:a", "128k"])
        .arg(out)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Command("ffmpeg audio extraction failed".into()))
    }
}
