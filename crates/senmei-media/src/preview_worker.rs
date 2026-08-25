//! Shared preview decode worker: one thread owns the `PreviewCache` (warm
//! streams + decode budget), coalesces last-frame-wins per input, and answers
//! with a raw frame. Transport-agnostic — Tauri and HTTP both drive it.

use std::path::PathBuf;
use std::sync::mpsc;

use crate::frame::Frame;
use crate::PreviewCache;

/// One decode request; the worker answers on `respond`.
struct PreviewRequest {
    input: String,
    position_ms: f64,
    respond: mpsc::Sender<Result<Frame, String>>,
}

/// Last-frame-wins: drain the queue, keep the newest request per input;
/// superseded callers get the newest frame (no stale decodes behind a slow one).
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

/// Serialized decodes, no lock: one thread owns the `PreviewCache`.
pub struct PreviewWorker {
    tx: mpsc::Sender<PreviewRequest>,
}

impl PreviewWorker {
    /// Spawn the worker; `ffmpeg` is the resolved binary (system/portable).
    pub fn new(ffmpeg: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel::<PreviewRequest>();
        std::thread::Builder::new()
            .name("preview-decode".into())
            .spawn(move || {
                let mut cache = PreviewCache::new(ffmpeg, Some(crate::PREVIEW_MAX_DIM));
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
        Self { tx }
    }

    /// Decode one frame (raw RGB24) at `position_ms` via the worker.
    pub fn frame(&self, input: &str, position_ms: f64) -> Result<Frame, String> {
        let (respond, recv) = mpsc::channel();
        self.tx
            .send(PreviewRequest {
                input: input.to_string(),
                position_ms,
                respond,
            })
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
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
        let pos = |v: &[PreviewRequest]| -> Vec<f64> { v.iter().map(|r| r.position_ms).collect() };
        assert_eq!(pos(&latest).iter().sum::<f64>(), 2400.0); // a@1500 + b@900
        assert_eq!(pos(&stale).iter().sum::<f64>(), 3500.0); // a@1000 + b@500 + a@2000
    }
}
