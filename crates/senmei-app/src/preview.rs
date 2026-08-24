//! Frame preview helpers: persistent decode streams + raw-frame delivery.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{mpsc, OnceLock};

use crate::store;

/// Preview decode budget (longest edge) — display-sized, keeps scrubbing cheap.
const PREVIEW_MAX_DIM: u32 = 1280;

/// One decode request; the worker answers on `respond`.
struct PreviewRequest {
    input: String,
    position_ms: f64,
    respond: mpsc::Sender<Result<senmei_media::Frame, String>>,
}

static PREVIEW_WORKER: OnceLock<mpsc::Sender<PreviewRequest>> = OnceLock::new();

/// Last-frame-wins: drain the worker queue and keep only the newest request
/// per input. Superseded requests are returned so the worker can answer them
/// with the newest frame — a fast scrub must not queue stale decodes behind a
/// slow one (upscaled results), and the superseded callers unblock instead of
/// waiting on positions nobody wants.
fn coalesce(
    first: PreviewRequest,
    rx: &mpsc::Receiver<PreviewRequest>,
) -> (Vec<PreviewRequest>, Vec<PreviewRequest>) {
    let mut latest = vec![first];
    let mut stale = Vec::new();
    let mut absorb = |req: PreviewRequest| match latest.iter_mut().find(|l| l.input == req.input) {
        Some(slot) => stale.push(std::mem::replace(slot, req)),
        None => latest.push(req),
    };
    while let Ok(next) = rx.try_recv() {
        absorb(next);
    }
    (latest, stale)
}

/// Lazily spawn the single preview-decode worker. It owns the `PreviewCache`
/// (warm decode streams = ring buffer) on one thread, so decodes are
/// serialized without a shared lock and no thread is spawned per request.
fn worker() -> &'static mpsc::Sender<PreviewRequest> {
    PREVIEW_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<PreviewRequest>();
        std::thread::Builder::new()
            .name("preview-decode".into())
            .spawn(move || {
                let ffmpeg = senmei_media::resolve(&store::data_dir());
                let mut cache = senmei_media::PreviewCache::new(ffmpeg, Some(PREVIEW_MAX_DIM));
                while let Ok(req) = rx.recv() {
                    let (latest, stale) = coalesce(req, &rx);
                    for r in &latest {
                        let res = cache.frame(&r.input, r.position_ms).map_err(|e| {
                            log::warn!("preview decode failed: {e}");
                            e.to_string()
                        });
                        for old in stale.iter().filter(|o| o.input == r.input) {
                            let _ = old.respond.send(res.clone());
                        }
                        let _ = r.respond.send(res);
                    }
                }
            })
            .expect("failed to spawn preview decode thread");
        tx
    })
}

/// A decoded preview frame: raw RGB24 bytes, base64-encoded for the IPC/JSON
/// transport. `width`/`height` let the frontend build an `ImageData` directly
/// (no `<img>`/PNG round-trip).
#[derive(Debug, Clone, serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    /// base64-encoded RGB24 pixels.
    pub data: String,
}

impl FrameData {
    fn from_frame(f: senmei_media::Frame) -> Self {
        use base64::Engine as _;
        Self {
            width: f.width,
            height: f.height,
            data: base64::engine::general_purpose::STANDARD.encode(f.data),
        }
    }
}

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

/// Decode one frame at `position_ms` as raw RGB24 (base64) for the preview
/// monitor. Sends the request to the single decode worker, which serves it
/// from its warm decode streams (one ffmpeg per file) and applies the preview
/// decode budget.
pub fn read_frame_inner(
    input: &str,
    position_ms: f64,
    _project_dir: Option<&str>,
) -> Result<FrameData, String> {
    let (respond, recv) = mpsc::channel();
    worker()
        .send(PreviewRequest {
            input: input.to_string(),
            position_ms,
            respond,
        })
        .map_err(|e| e.to_string())?;
    let frame = recv.recv().map_err(|e| e.to_string())??;
    Ok(FrameData::from_frame(frame))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coalescing keeps only the newest position per input and returns the
    /// superseded requests for answering with the newest frame (last-frame-
    /// wins): a@1000, a@2000, a@1500 → keep a@1500; b@500, b@900 → keep b@900.
    #[test]
    fn coalesce_keeps_newest_position_per_input() {
        let (tx, rx) = mpsc::channel::<PreviewRequest>();
        let (rt, _rr) = mpsc::channel();
        let mk = |input: &str, pos: f64| PreviewRequest {
            input: input.to_string(),
            position_ms: pos,
            respond: rt.clone(),
        };
        tx.send(mk("a.mp4", 1000.0)).unwrap();
        tx.send(mk("b.mp4", 500.0)).unwrap();
        tx.send(mk("a.mp4", 2000.0)).unwrap();
        tx.send(mk("b.mp4", 900.0)).unwrap();
        tx.send(mk("a.mp4", 1500.0)).unwrap();

        let first = rx.recv().unwrap();
        let (latest, stale) = coalesce(first, &rx);
        let pos = |v: &[PreviewRequest]| -> Vec<f64> {
            v.iter().map(|r| r.position_ms).collect()
        };
        assert_eq!(pos(&latest).iter().sum::<f64>(), 2400.0); // a@1500 + b@900
        assert_eq!(pos(&stale).iter().sum::<f64>(), 3500.0); // a@1000 + b@500 + a@2000
    }
}

/// Extract the source audio once as stereo AAC for the native preview player —
/// any source audio codec (e.g. AC3 in anime files) is decoded by our FFmpeg
/// and re-encoded to AAC (small + seekable in rodio). One active track at a
/// time; stale tracks (incl. old .wav/.flac/.mp3/.webm/.m4a) are dropped when
/// a new one is extracted.
pub fn extract_audio_inner(input: &str, project_dir: Option<&str>) -> Result<String, String> {
    let dir = preview_dir(project_dir);
    let ns = frame_ns(input);
    let path = dir.join(format!("audio_{ns}.aac"));
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
                && (name.ends_with(".aac")
                    || name.ends_with(".wav")
                    || name.ends_with(".flac")
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
    // leaves a partial track at the cached path. The .aac suffix lets ffmpeg
    // infer the ADTS muxer (it can't from a bare .tmp).
    let tmp = dir.join(format!("audio_{ns}.tmp.aac"));
    let _ = std::fs::remove_file(&tmp);
    senmei_media::extract_audio(&ffmpeg, std::path::Path::new(input), &tmp).map_err(|e| {
        log::warn!("audio extraction failed: {e}");
        e.to_string()
    })?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}
