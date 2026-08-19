//! Frame preview helpers: persistent decode streams + PNG writes.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::store;

static PREVIEW_CACHE: OnceLock<Mutex<Option<senmei_media::PreviewCache>>> = OnceLock::new();
static PREVIEW_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn probe_video_inner(input: &str) -> Result<senmei_media::VideoInfo, String> {
    let ffmpeg = senmei_media::resolve(&store::data_dir());
    let ffprobe = senmei_media::ffprobe_next_to(&ffmpeg);
    senmei_media::probe(&ffprobe, std::path::Path::new(input)).map_err(|e| {
        log::warn!("probe_video failed: {e}");
        e.to_string()
    })
}

/// Short stable namespace for one input file, so original/result/compare frames
/// never share a prune bucket or filename.
fn frame_ns(input: &str) -> String {
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Preview scratch dir: under the project (`preview/`) when one is open, else
/// the app data dir; a non-writable project dir falls back to the data dir.
fn preview_dir(project_dir: Option<&str>) -> std::path::PathBuf {
    project_dir
        .and_then(|p| {
            let d = std::path::Path::new(p).join("preview");
            std::fs::create_dir_all(&d).ok().map(|_| d)
        })
        .unwrap_or_else(|| {
            let d = store::data_dir().join("preview");
            let _ = std::fs::create_dir_all(&d);
            d
        })
}

/// Extract one frame at `position_ms` as a PNG file and return its path. Uses
/// a persistent decode stream (one ffmpeg per file) so playback reads frames
/// from the pipe instead of spawning a process per frame. Frames are written
/// under the project (`preview/`) when one is open, else the app data dir;
/// a non-writable project dir falls back to the data dir.
pub fn read_frame_inner(
    input: &str,
    position_ms: f64,
    project_dir: Option<&str>,
) -> Result<String, String> {
    let dir = preview_dir(project_dir);
    let ns = frame_ns(input);
    prune_preview_frames(&dir, &ns, 60);

    let ffmpeg = senmei_media::resolve(&store::data_dir());
    // Decode under the lock (fast pipe read) but encode outside it, so the two
    // compare sides don't serialize their (slow) PNG encode behind each other.
    let frame = {
        let mut cache = PREVIEW_CACHE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|e| e.to_string())?;
        if cache.is_none() {
            *cache = Some(senmei_media::PreviewCache::new(ffmpeg));
        }
        cache
            .as_mut()
            .unwrap()
            .frame(input, position_ms)
            .map_err(|e| {
                log::warn!("preview decode failed: {e}");
                e.to_string()
            })?
    };
    let png = senmei_media::encode_png(frame.width, frame.height, &frame.data)
        .map_err(|e| e.to_string())?;

    let seq = PREVIEW_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("frame_{ns}_{seq:08}.png"));
    std::fs::write(&path, &png).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Best-effort cap on leftover preview frames for one namespace (keep the
/// newest `keep`). Zero-padded counters make name order equal to write order.
fn prune_preview_frames(dir: &std::path::Path, ns: &str, keep: usize) {
    let prefix = format!("frame_{ns}_");
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut old: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().starts_with(&prefix))
                    .unwrap_or(false)
            })
            .collect();
        old.sort();
        for p in old.iter().take(old.len().saturating_sub(keep)) {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Extract the source audio once as AAC (M4A) for the preview `<audio>` — the
/// webview can't always decode the source's audio codec (e.g. AC3 in anime
/// files), but every webview plays AAC/M4A. One active track at a time; stale
/// tracks are dropped when a new one is extracted.
pub fn extract_audio_inner(input: &str, project_dir: Option<&str>) -> Result<String, String> {
    let dir = preview_dir(project_dir);
    let ns = frame_ns(input);
    let path = dir.join(format!("audio_{ns}.m4a"));
    if path.exists() {
        return Ok(path.to_string_lossy().into_owned());
    }
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("audio_") && name.ends_with(".m4a") && e.path() != path {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let ffmpeg = senmei_media::resolve(&store::data_dir());
    senmei_media::extract_audio(&ffmpeg, std::path::Path::new(input), &path).map_err(|e| {
        log::warn!("audio extraction failed: {e}");
        e.to_string()
    })?;
    Ok(path.to_string_lossy().into_owned())
}
