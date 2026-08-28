//! One-frame JPEG thumbnail for the media library tiles. ffmpeg decodes a
//! single frame (a quarter in, capped at 1s) and re-encodes it as MJPEG.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::{Error, Result, VideoInfo};

/// JPEG bytes plus the source probe. The probe is needed anyway to pick the
/// seek timestamp, so returning it lets the caller skip a second ffprobe
/// spawn (the MediaLibrary tile renders `WxH · codec` from here).
pub struct Thumbnail {
    pub jpeg: Vec<u8>,
    pub info: VideoInfo,
}

/// Best timestamp to thumbnail, in seconds: a quarter in (past any title
/// card) but capped at 1s so the seek stays cheap on long files.
fn at_seconds(duration: f64) -> f64 {
    if duration <= 0.0 {
        0.0
    } else {
        (duration * 0.25).min(1.0)
    }
}

/// Extract a JPEG thumbnail of `input`. `max_w` caps the width; the height
/// follows the source aspect (even value keeps the scale filter happy).
pub fn thumbnail(ffmpeg: &Path, input: &Path, max_w: u32) -> Result<Thumbnail> {
    let info = crate::probe(&crate::ffprobe_next_to(ffmpeg), input)?;
    let at = at_seconds(info.duration);
    let mut child = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{at:.3}"), "-i"])
        .arg(input)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &format!("scale={max_w}:-2"),
            "-q:v",
            "5",
            "-f",
            "mjpeg",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Command(format!("ffmpeg spawn failed: {e}")))?;

    let mut buf = Vec::new();
    child
        .stdout
        .take()
        .and_then(|mut o| o.read_to_end(&mut buf).ok())
        .ok_or_else(|| Error::Command("ffmpeg thumbnail read failed".into()))?;
    let mut err = Vec::new();
    child.stderr.take().and_then(|mut o| o.read_to_end(&mut err).ok());
    let status = child
        .wait()
        .map_err(|e| Error::Command(format!("ffmpeg wait failed: {e}")))?;
    if !status.success() {
        return Err(Error::Command(format!(
            "ffmpeg thumbnail failed: {}",
            String::from_utf8_lossy(&err)
        )));
    }
    if buf.is_empty() {
        return Err(Error::Command("ffmpeg produced no thumbnail".into()));
    }
    Ok(Thumbnail { jpeg: buf, info })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    fn ff() -> Option<PathBuf> {
        std::env::var("SENMEI_FFMPEG")
            .ok()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
    }

    /// Thumbnailing a generated clip yields a non-empty JPEG.
    #[test]
    #[ignore = "needs SENMEI_FFMPEG"]
    fn jpeg_magic_bytes() {
        let Some(ff) = ff() else {
            eprintln!("SENMEI_FFMPEG not set, skipping");
            return;
        };
        let dir = std::env::temp_dir().join("senmei-thumb-test");
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("clip.mp4");
        let _ = Command::new(&ff)
            .args(["-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x180:rate=24", "-t", "1", "-pix_fmt", "yuv420p"])
            .arg(&clip)
            .status();
        let thumb = thumbnail(&ff, &clip, 160).expect("thumbnail extract");
        assert!(thumb.jpeg.len() > 100, "non-trivial JPEG");
        assert_eq!(&thumb.jpeg[..2], &[0xff, 0xd8], "JPEG magic");
        assert!(thumb.info.width > 0, "probe returned by thumbnail");
        let _ = std::fs::remove_file(&clip);
    }
}
