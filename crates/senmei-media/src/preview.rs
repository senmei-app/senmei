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

/// Extract the source audio track as stereo AAC for the native preview player
/// (rodio). Any source codec (AC3/FLAC/Opus/…) is decoded by our FFmpeg and
/// re-encoded to AAC (~192 kbps). The native rodio decoders are not seekable
/// (FLAC/Vorbis/MP3), so raw WAV would be the only seekable lossless choice —
/// but a long 5.1 source turns into 3+ GB. AAC is small and seeks via rodio's
/// symphonia decoder.
pub fn extract_audio(
    ffmpeg: &std::path::Path,
    input: &std::path::Path,
    out: &std::path::Path,
) -> Result<()> {
    let status = std::process::Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "2", "-c:a", "aac", "-b:a", "192k"])
        .arg(out)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Command("ffmpeg audio extraction failed".into()))
    }
}
