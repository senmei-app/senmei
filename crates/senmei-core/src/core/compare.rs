//! Render quality metrics: PSNR/SSIM comparison against the source.

use std::path::Path;
use super::ffmpeg;

/// Parse the last occurrence of `key` in ffmpeg's stderr summary lines
/// (PSNR `average:`, SSIM `All:`).
fn parse_after(stderr: &str, key: &str) -> Option<f64> {
    stderr.lines().rev().find_map(|l| {
        let rest = l.split_once(key)?.1.trim_start();
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        num.parse().ok()
    })
}

/// Run one FFmpeg metric filter (psnr/ssim) between two clips, scaling the
/// rendered clip back to the original resolution. Returns the parsed summary.
fn run_metric(
    ff: &Path,
    rendered: &str,
    original: &str,
    scale: &str,
    filter: &str,
    key: &str,
) -> Result<Option<f64>, String> {
    let lavfi = format!("[0:v]{scale}format=yuv420p[s];[1:v]format=yuv420p[r];[s][r]{filter}");
    let out = std::process::Command::new(ff)
        .args([
            "-hide_banner",
            "-i",
            rendered,
            "-i",
            original,
            "-lavfi",
            &lavfi,
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg {filter} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(parse_after(&String::from_utf8_lossy(&out.stderr), key))
}

/// Compare a rendered sample against its source: PSNR (dB) + SSIM, both on the
/// original resolution (the rendered clip is scaled back down). VMAF is null —
/// it needs a libvmaf FFmpeg build (§16.5).
pub fn compare_sample(original: &str, rendered: &str) -> Result<serde_json::Value, String> {
    let ff = ffmpeg();
    let ffprobe = senmei_media::ffprobe_next_to(&ff);
    let orig = senmei_media::probe(&ffprobe, Path::new(original)).map_err(|e| e.to_string())?;
    let rend = senmei_media::probe(&ffprobe, Path::new(rendered)).map_err(|e| e.to_string())?;

    let scale = if (orig.width, orig.height) != (rend.width, rend.height) {
        format!("scale={}:{}:flags=bicubic,", orig.width, orig.height)
    } else {
        String::new()
    };

    let psnr_db = run_metric(&ff, rendered, original, &scale, "psnr", "average:")?;
    let ssim = run_metric(&ff, rendered, original, &scale, "ssim", "All:")?;

    Ok(serde_json::json!({
        "original": { "path": original, "width": orig.width, "height": orig.height },
        "rendered": { "path": rendered, "width": rend.width, "height": rend.height },
        "psnrDb": psnr_db,
        "ssim": ssim,
        "vmaf": null,
        "note": "PSNR/SSIM on the original resolution (rendered downscaled); VMAF needs a libvmaf FFmpeg build",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metric_summaries() {
        let psnr =
            "[Parsed_psnr_0 @ 0x55] PSNR y:37.0 u:41.0 v:41.0 average:38.234 min:36.1 max:40.2";
        assert_eq!(parse_after(psnr, "average:"), Some(38.234));
        let ssim = "[Parsed_ssim_0 @ 0x55] SSIM Y:0.98 (12.0) U:0.97 (11.0) V:0.96 (10.0) All:0.981234 (12.3)";
        assert_eq!(parse_after(ssim, "All:"), Some(0.981234));
        assert_eq!(parse_after("All:0.1 (1)\nAll:0.9 (2)", "All:"), Some(0.9));
    }
}
