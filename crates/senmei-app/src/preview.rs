//! Frame preview helpers: persistent decode streams + PNG writes.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::store;

static PREVIEW_CACHE: OnceLock<Mutex<Option<senmei_media::PreviewCache>>> = OnceLock::new();
static PREVIEW_SEQ: AtomicU64 = AtomicU64::new(0);
/// Preview decode budget (longest edge) — display-sized, keeps scrubbing cheap.
const PREVIEW_MAX_DIM: u32 = 1280;

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

    let ffmpeg = senmei_media::resolve(&store::data_dir());
    // Decode under the lock (fast pipe read) but encode outside it, so the two
    // compare sides don't serialize their (slow) PNG encode behind each other.
    let frame = {
        let mut cache = PREVIEW_CACHE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|e| e.to_string())?;
        if cache.is_none() {
            *cache = Some(senmei_media::PreviewCache::new(ffmpeg, Some(PREVIEW_MAX_DIM)));
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
    // Stable filename + atomic overwrite: the webview re-fetches on a new query
    // string (see Monitor) and never hits a pruned/mid-write file.
    let path = dir.join(format!("frame_{ns}.png"));
    let tmp = dir.join(format!("frame_{ns}.{seq}.tmp"));
    std::fs::write(&tmp, &png).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    cap_preview_dir(&dir);
    Ok(path.to_string_lossy().into_owned())
}

/// Keep the preview dir bounded: drop the oldest files (any namespace) beyond
/// a cap. Frames now use a stable name, so only cross-file leftovers remain.
fn cap_preview_dir(dir: &std::path::Path) {
    const MAX_FILES: usize = 400;
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    if files.len() <= MAX_FILES {
        return;
    }
    files.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    for p in files.iter().take(files.len() - MAX_FILES) {
        let _ = std::fs::remove_file(p);
    }
}

/// Extract the source audio once as FLAC for the native preview player — any
/// source audio codec (e.g. AC3 in anime files) is decoded by our FFmpeg and
/// re-encoded losslessly, so rodio never sees an exotic codec. One active
/// track at a time; stale tracks (incl. old .mp3/.webm/.m4a) are dropped when
/// a new one is extracted.
pub fn extract_audio_inner(input: &str, project_dir: Option<&str>) -> Result<String, String> {
    let dir = preview_dir(project_dir);
    let ns = frame_ns(input);
    let path = dir.join(format!("audio_{ns}.flac"));
    // Cache only complete tracks; a failed run must not leave a 0-byte file
    // that later looks "done".
    if path.exists() && std::fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(path.to_string_lossy().into_owned());
    }
    let _ = std::fs::remove_file(&path);
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let stale = name.starts_with("audio_")
                && (name.ends_with(".flac")
                    || name.ends_with(".mp3")
                    || name.ends_with(".webm")
                    || name.ends_with(".m4a"))
                && e.path() != path;
            if stale {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let ffmpeg = senmei_media::resolve(&store::data_dir());
    // Extract to a temp name and rename only on success so a failure never
    // leaves a partial track at the cached path. The .flac suffix lets ffmpeg
    // infer the muxer (it can't from a bare .tmp).
    let tmp = dir.join(format!("audio_{ns}.tmp.flac"));
    let _ = std::fs::remove_file(&tmp);
    senmei_media::extract_audio(&ffmpeg, std::path::Path::new(input), &tmp).map_err(|e| {
        log::warn!("audio extraction failed: {e}");
        e.to_string()
    })?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}
