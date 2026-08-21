//! HTTP adapter over the core service — serves the full web UI + REST API.
//! Same license/confirm gates as MCP (they live in `core`).

use axum::{
    Json, Router,
    body::Body,
    http::{Request, Response, StatusCode, header},
    routing::{get, post},
};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};

use crate::core;

/// The built web UI (packages/app/dist), embedded at compile time. Empty when
/// the frontend hasn't been built yet (bare `cargo check`).
#[derive(RustEmbed)]
#[folder = "../../packages/app/dist"]
pub struct WebUi;

/// Serve the embedded UI; unknown paths fall back to `index.html` (SPA), but
/// `/api/*` 404s so unmatched REST calls don't return HTML.
async fn embedded_fallback(req: Request<Body>) -> Response<Body> {
    let path = req.uri().path().trim_start_matches('/');
    if path.starts_with("api/") {
        return not_found();
    }
    let file = if path.is_empty() {
        WebUi::get("index.html")
    } else {
        WebUi::get(path).or_else(|| WebUi::get("index.html"))
    };
    match file {
        Some(f) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, f.metadata.mimetype())
            .body(Body::from(f.data.into_owned()))
            .unwrap(),
        None => not_found(),
    }
}

fn not_found() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Body::empty())
        .unwrap()
}

#[derive(Deserialize)]
struct ProbeParams {
    input: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameParams {
    input: String,
    position_ms: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DownloadParams {
    model_id: String,
}

#[derive(Deserialize)]
struct CompareParams {
    original: String,
    rendered: String,
}

type ApiResult = (StatusCode, Json<serde_json::Value>);

fn json_ok<T: Serialize>(v: &T) -> ApiResult {
    (StatusCode::OK, Json(serde_json::to_value(v).unwrap_or_default()))
}

fn json_err(status: StatusCode, msg: impl Into<String>) -> ApiResult {
    (status, Json(serde_json::json!({ "error": msg.into() })))
}

async fn models() -> ApiResult {
    json_ok(&core::list_models())
}

async fn settings_schema() -> ApiResult {
    json_ok(&core::settings_schema())
}

async fn ffmpeg_status() -> ApiResult {
    json_ok(&core::ffmpeg_status())
}

async fn backend_info() -> ApiResult {
    json_ok(&senmei_ml::backend_info())
}

async fn probe(Json(p): Json<ProbeParams>) -> ApiResult {
    match core::probe_video(&p.input) {
        Ok(info) => json_ok(&info),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn frame(Json(p): Json<FrameParams>) -> ApiResult {
    match core::frame_png(&p.input, p.position_ms) {
        Ok(data) => json_ok(&serde_json::json!({ "data": data, "mime": "image/png" })),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn compare(Json(p): Json<CompareParams>) -> ApiResult {
    match core::compare_sample(&p.original, &p.rendered) {
        Ok(metrics) => json_ok(&metrics),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn download_model(Json(p): Json<DownloadParams>) -> ApiResult {
    #[cfg(feature = "render")]
    {
        return match tokio::task::spawn_blocking(move || core::download_model(&p.model_id)).await
        {
            Ok(Ok(path)) => json_ok(&serde_json::json!({ "bpk": path })),
            Ok(Err(e)) => json_err(StatusCode::BAD_REQUEST, e),
            Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("join failed: {e}")),
        };
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

async fn render_start(Json(cfg): Json<core::RenderConfig>) -> ApiResult {
    #[cfg(feature = "render")]
    {
        return match core::propose_render(cfg).and_then(|_| core::confirm_render()) {
            Ok(msg) => json_ok(&serde_json::json!({ "started": msg })),
            Err(e) => json_err(StatusCode::BAD_REQUEST, e),
        };
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

async fn render_status() -> ApiResult {
    #[cfg(feature = "render")]
    {
        return json_ok(&core::render_status());
    }
    #[cfg(not(feature = "render"))]
    json_err(StatusCode::SERVICE_UNAVAILABLE, "render not compiled in")
}

async fn render_cancel() -> ApiResult {
    #[cfg(feature = "render")]
    {
        core::cancel_render();
        return json_ok(&serde_json::json!({ "cancelled": true }));
    }
    #[cfg(not(feature = "render"))]
    json_err(StatusCode::SERVICE_UNAVAILABLE, "render not compiled in")
}

/// Build the HTTP router: REST API + optional static UI (ServeDir fallback).
pub fn router(web_dir: Option<std::path::PathBuf>) -> Router {
    let api = Router::new()
        .route("/api/health", get(|| async { "ok" }))
        .route("/api/models", get(models))
        .route("/api/settings-schema", get(settings_schema))
        .route("/api/ffmpeg", get(ffmpeg_status))
        .route("/api/backend-info", get(backend_info))
        .route("/api/probe", post(probe))
        .route("/api/frame", post(frame))
        .route("/api/compare", post(compare))
        .route("/api/download-model", post(download_model))
        .route("/api/render", post(render_start))
        .route("/api/render/status", get(render_status))
        .route("/api/render/cancel", post(render_cancel));

    // Permissive CORS for the Vite dev server (localhost:1420 → :8765).
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    match web_dir {
        Some(dir) => {
            let serve = tower_http::services::ServeDir::new(dir)
                .append_index_html_on_directories(true);
            api.layer(cors).fallback_service(serve)
        }
        // No dir given: serve the UI embedded in the binary (SPA fallback).
        None => api.layer(cors).fallback(embedded_fallback),
    }
}
