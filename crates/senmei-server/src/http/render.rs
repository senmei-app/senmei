//! Render/queue HTTP handlers (download, start, status, cancel).

use super::*;

#[derive(Deserialize)]
pub(super) struct DownloadParams {
    pub(super) model_id: String,
}

pub(super) async fn download_model(Json(p): Json<DownloadParams>) -> ApiResult {
    #[cfg(feature = "render")]
    {
        match tokio::task::spawn_blocking(move || core::download_model(&p.model_id, |_, _| {}))
            .await
        {
            Ok(Ok(path)) => json_ok(&serde_json::json!({ "bpk": path })),
            Ok(Err(e)) => json_err(StatusCode::BAD_REQUEST, e),
            Err(e) => json_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join failed: {e}"),
            ),
        }
    }
    #[cfg(not(feature = "render"))]
    {
        let _ = p.model_id;
        json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "render not compiled in (build with --features render,http)",
        )
    }
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
        // Mirror the desktop's config log so HTTP renders are auditable.
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
        let _ = cfg;
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
