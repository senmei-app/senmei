//! Render/queue HTTP handlers (download, start, status, cancel).

use super::*;
use futures_core::Stream;
use tokio_stream::StreamExt;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct DownloadParams {
    pub(super) model_id: String,
}

enum DownloadMsg {
    Progress { downloaded: u64, total: u64 },
    Done { bpk: String },
    Error { error: String },
}

/// SSE download: streams progress events, then a final `done` or `error` event.
#[cfg(feature = "render")]
pub(super) async fn download_model(
    Json(p): Json<DownloadParams>,
) -> axum::response::sse::Sse<
    impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::{Event, KeepAlive};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let (tx, rx) = mpsc::channel::<DownloadMsg>(32);

    tokio::task::spawn_blocking(move || {
        let result = core::download_model(&p.model_id, |downloaded, total| {
            // Err means the client disconnected — the task will end naturally
            // when the next send fails or the download completes.
            let _ = tx.blocking_send(DownloadMsg::Progress { downloaded, total });
        });
        match result {
            Ok(path) => {
                let _ = tx.blocking_send(DownloadMsg::Done { bpk: path });
            }
            Err(e) => {
                let _ = tx.blocking_send(DownloadMsg::Error { error: e });
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|msg| {
        let event = match msg {
            DownloadMsg::Progress { downloaded, total } => Event::default()
                .event("progress")
                .json_data(serde_json::json!({ "downloaded": downloaded, "total": total })),
            DownloadMsg::Done { bpk } => Event::default()
                .event("done")
                .json_data(serde_json::json!({ "bpk": bpk })),
            DownloadMsg::Error { error } => Event::default()
                .event("error")
                .json_data(serde_json::json!({ "error": error })),
        };
        Ok(event.unwrap_or_else(|_| Event::default().event("error").data("Serialization error")))
    });

    axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(not(feature = "render"))]
pub(super) async fn download_model(Json(p): Json<DownloadParams>) -> ApiResult {
    let _ = p.model_id;
    json_err(
        StatusCode::SERVICE_UNAVAILABLE,
        "render not compiled in (build with --features render,http)",
    )
}

pub(super) async fn render_start(
    State(state): State<AppState>,
    Json(cfg): Json<core::RenderConfig>,
) -> ApiResult {
    #[cfg(feature = "render")]
    {
        if !is_allowed(&state, Path::new(&cfg.input)) {
            return json_err(StatusCode::BAD_REQUEST, "input not opened");
        }
        register_parent(&state, Path::new(&cfg.output));
        log::info!(
            "http render start: {} -> {} (config {cfg:?})",
            cfg.input,
            cfg.output
        );
        match core::propose_render(cfg).and_then(|_| core::confirm_render()) {
            Ok(msg) => json_ok(&serde_json::json!({ "started": msg })),
            Err(e) => json_err(StatusCode::BAD_REQUEST, e),
        }
    }
    #[cfg(not(feature = "render"))]
    {
        let _ = (cfg, state);
        json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "render not compiled in (build with --features render,http)",
        )
    }
}

pub(super) async fn render_status() -> ApiResult {
    #[cfg(feature = "render")]
    {
        json_ok(&core::render_status())
    }
    #[cfg(not(feature = "render"))]
    json_err(StatusCode::SERVICE_UNAVAILABLE, "render not compiled in")
}

pub(super) async fn render_cancel() -> ApiResult {
    #[cfg(feature = "render")]
    {
        core::cancel_render();
        json_ok(&serde_json::json!({ "cancelled": true }))
    }
    #[cfg(not(feature = "render"))]
    json_err(StatusCode::SERVICE_UNAVAILABLE, "render not compiled in")
}
