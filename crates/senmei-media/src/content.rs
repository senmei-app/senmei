//! Content classification for pipeline suggestions (anime vs live-action).
//! Rough heuristic: anime is cel-shaded — large flat color regions with clean
//! edges; live-action is textured and noisy. Two sampled frames are downscaled
//! to grayscale and scored by flatness (luma variance) and edge energy
//! (Laplacian). Undecidable inputs default to live-action.

use std::io::Read;
use std::path::Path;
use std::process::Stdio;

/// Luma variance below this counts as "flat" (anime cel shading).
const FLAT_VAR: f64 = 2000.0;
/// Laplacian energy below this counts as "clean" (no film grain/noise).
const CLEAN_EDGE: f64 = 2000.0;
/// Laplacian energy below this counts as "blurry" for live-action.
const BLURRY_EDGE: f64 = 1000.0;

/// Rough anime/live classification. `true` = likely anime.
pub fn is_anime(ffmpeg: &Path, input: &Path, duration_ms: u64) -> bool {
    let mut flat = 0;
    let mut clean = 0;
    let mut n = 0;
    for frac in [0.3, 0.7] {
        if let Some((gv, lv)) = frame_stats(ffmpeg, input, duration_ms as f64 * frac) {
            if gv < FLAT_VAR {
                flat += 1;
            }
            if lv < CLEAN_EDGE {
                clean += 1;
            }
            n += 1;
        }
    }
    n > 0 && flat == n && clean == n
}

/// Rough blurry classification. `true` = likely blurry (very low edge energy).
pub fn is_blurry(ffmpeg: &Path, input: &Path, duration_ms: u64) -> bool {
    let mut blurry = 0;
    let mut n = 0;
    for frac in [0.3, 0.7] {
        if let Some((_gv, lv)) = frame_stats(ffmpeg, input, duration_ms as f64 * frac) {
            if lv < BLURRY_EDGE {
                blurry += 1;
            }
            n += 1;
        }
    }
    n > 0 && blurry == n
}

/// Downscale one frame to grayscale; returns (luma variance, Laplacian energy)
/// over the 64×N sample.
fn frame_stats(ffmpeg: &Path, input: &Path, at_ms: f64) -> Option<(f64, f64)> {
    let mut child = crate::process::hidden(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            &format!("{:.3}", at_ms / 1000.0),
            "-i",
        ])
        .arg(input)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=64:-1,format=gray",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut buf = Vec::new();
    // Reap the child — without wait() every probe leaves a zombie behind.
    let read = child
        .stdout
        .take()
        .and_then(|mut o| o.read_to_end(&mut buf).ok());
    let _ = child.wait();
    read?;
    let w = 64usize;
    let h = buf.len() / w;
    if h < 3 || buf.len() < w * h {
        return None;
    }

    let mut mean = 0.0;
    for &v in &buf {
        mean += v as f64;
    }
    mean /= buf.len() as f64;

    let mut gv = 0.0;
    for &v in &buf {
        let d = v as f64 - mean;
        gv += d * d;
    }
    gv /= buf.len() as f64;

    let mut lv = 0.0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let lap = 4.0 * buf[i] as f64
                - buf[i - w] as f64
                - buf[i + w] as f64
                - buf[i - 1] as f64
                - buf[i + 1] as f64;
            lv += lap * lap;
        }
    }
    lv /= ((h - 2) * (w - 2)) as f64;
    Some((gv, lv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn clip(ff: &Path, out: &Path, src: &str) {
        let _ = Command::new(ff)
            .args([
                "-y", "-f", "lavfi", "-i", src, "-t", "1", "-pix_fmt", "yuv420p",
            ])
            .arg(out)
            .status();
    }

    /// Flat color → anime-like; heavy noise → live-action-like.
    #[test]
    #[ignore = "needs SENMEI_FFMPEG"]
    fn flat_anime_versus_noisy_live() {
        let Some(ff) = std::env::var("SENMEI_FFMPEG")
            .ok()
            .filter(|p| !p.is_empty())
        else {
            eprintln!("SENMEI_FFMPEG not set, skipping");
            return;
        };
        let ff = Path::new(&ff);
        let dir = std::env::temp_dir().join("senmei-anime-test");
        std::fs::create_dir_all(&dir).unwrap();
        let anime = dir.join("anime.mp4");
        let live = dir.join("live.mp4");
        clip(
            ff,
            &anime,
            "color=c=0x6688aa:duration=1:size=320x180:rate=24",
        );
        clip(
            ff,
            &live,
            "testsrc=duration=1:size=320x180:rate=24,noise=alls=20:allf=t",
        );
        assert!(
            is_anime(ff, &anime, 1000),
            "flat color should look like anime"
        );
        assert!(!is_anime(ff, &live, 1000), "noisy testsrc should look live");
        
        // Test blurry
        let blurry_live = dir.join("blurry_live.mp4");
        clip(
            ff,
            &blurry_live,
            "testsrc=duration=1:size=320x180:rate=24,gblur=sigma=10",
        );
        assert!(
            is_blurry(ff, &blurry_live, 1000),
            "blurred testsrc should look blurry"
        );
        assert!(
            !is_blurry(ff, &live, 1000),
            "noisy testsrc should not look blurry"
        );

        let _ = std::fs::remove_file(&anime);
        let _ = std::fs::remove_file(&live);
        let _ = std::fs::remove_file(&blurry_live);
    }
}
