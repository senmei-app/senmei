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

/// Extract the source audio track as WAV (lossless PCM) for the native
/// preview player (rodio). Any source codec (AC3/FLAC/Opus/…) is decoded by
/// our FFmpeg and re-encoded losslessly, so rodio never sees an exotic codec.
/// WAV (pcm_s16le) is required: rodio's FLAC decoder is not seekable, so
/// scrubbing/play-start silently stayed at position 0.
pub fn extract_audio(
    ffmpeg: &std::path::Path,
    input: &std::path::Path,
    out: &std::path::Path,
) -> Result<()> {
    let status = std::process::Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vn", "-c:a", "pcm_s16le"])
        .arg(out)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Command("ffmpeg audio extraction failed".into()))
    }
}
