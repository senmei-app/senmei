//! Frame preview helpers: persistent decode streams + raw-frame delivery.

use std::sync::OnceLock;

use tauri::ipc::{InvokeResponseBody, IpcResponse};

use crate::store;

/// Single decode worker owns the `PreviewCache`: serialized decodes, no lock.
fn worker() -> &'static senmei_media::PreviewWorker {
    static PREVIEW_WORKER: OnceLock<senmei_media::PreviewWorker> = OnceLock::new();
    PREVIEW_WORKER.get_or_init(|| {
        let ffmpeg = senmei_media::resolve(&store::data_dir());
        senmei_media::PreviewWorker::new(ffmpeg)
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

/// Raw RGB24 for the frame channel — `IpcResponse::Raw` = ArrayBuffer (no
/// base64); not `Serialize` so the blanket JSON impl doesn't apply. Specta
/// can't type `ArrayBuffer`, so the frontend casts the channel.
#[derive(Debug, Clone, specta::Type)]
pub struct FramePixels(pub Vec<u8>);

impl IpcResponse for FramePixels {
    fn body(self) -> tauri::Result<InvokeResponseBody> {
        Ok(InvokeResponseBody::Raw(self.0))
    }
}

/// Decode one frame as raw RGB24 via the worker (warm streams + decode budget).
pub fn read_frame_inner(
    input: &str,
    position_ms: f64,
    _project_dir: Option<&str>,
) -> Result<senmei_media::Frame, String> {
    worker().frame(input, position_ms)
}
