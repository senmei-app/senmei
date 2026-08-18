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
