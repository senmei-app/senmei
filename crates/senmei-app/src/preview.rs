//! Frame preview helpers: persistent decode streams + raw-frame delivery.

use std::sync::{mpsc, OnceLock};

use tauri::ipc::{InvokeResponseBody, IpcResponse};

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

/// Width/height of a decoded preview frame, delivered as JSON on the meta
/// channel ahead of the raw pixels.
#[derive(Debug, Clone, serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct FrameMeta {
    pub width: u32,
    pub height: u32,
}

/// Raw RGB24 preview pixels for the frame channel. `IpcResponse` delivers
/// them as an `ArrayBuffer` (no base64/JSON round-trip); deliberately not
/// `Serialize`, so the blanket JSON `IpcResponse` impl doesn't apply. Specta
/// can't express `ArrayBuffer` (it types `Vec<u8>` as `number[]`), so the
/// frontend wrapper casts the channel to the runtime type.
#[derive(Debug, Clone, specta::Type)]
pub struct FramePixels(pub Vec<u8>);

impl IpcResponse for FramePixels {
    fn body(self) -> tauri::Result<InvokeResponseBody> {
        Ok(InvokeResponseBody::Raw(self.0))
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

/// Decode one frame at `position_ms` as raw RGB24 for the preview monitor.
/// Sends the request to the single decode worker, which serves it from its
/// warm decode streams (one ffmpeg per file) and applies the preview decode
/// budget. Transport-agnostic: the caller frames the bytes (Tauri raw Channel
/// vs HTTP base64).
pub fn read_frame_inner(
    input: &str,
    position_ms: f64,
    _project_dir: Option<&str>,
) -> Result<senmei_media::Frame, String> {
    let (respond, recv) = mpsc::channel();
    worker()
        .send(PreviewRequest {
            input: input.to_string(),
            position_ms,
            respond,
        })
        .map_err(|e| e.to_string())?;
    recv.recv().map_err(|e| e.to_string())?
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


