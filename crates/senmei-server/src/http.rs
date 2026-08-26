//! HTTP adapter over the core service — serves the full web UI + REST API.
//! Same license/confirm gates as MCP (they live in `core`).

use std::sync::OnceLock;
use std::time::SystemTime;

use axum::{
    body::Body,
    extract::Query,
    http::{header, Method, Request, Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use tower::ServiceExt;

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

/// The localhost REST API serves caller-supplied paths (`stream`/`audio`/
/// `frame`/`probe`). Restrict to real media files so a cross-origin page can't
/// read arbitrary local files (CORS is locked down too — see `router`).
fn media_path(p: &std::path::Path) -> bool {
    p.is_file()
        && p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            matches!(
                e,
                "mp4" | "m4v" | "mkv" | "webm" | "mov" | "avi" | "ts" | "m2ts" | "mts"
                    | "flv" | "wmv" | "mpg" | "mpeg" | "vob" | "3gp" | "f4v" | "ogv"
                    | "mp3" | "flac" | "ogg" | "oga" | "opus" | "wav" | "aac" | "m4a"
                    | "wma" | "ac3" | "ape"
            )
        })
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
struct ScanParams {
    dir: String,
}

#[derive(Deserialize)]
struct CompareParams {
    original: String,
    rendered: String,
}

type ApiResult = (StatusCode, Json<serde_json::Value>);

fn json_ok<T: Serialize>(v: &T) -> ApiResult {
    let value = serde_json::to_value(v).unwrap_or_else(|e| {
        log::error!("http serialization failed: {e}");
        serde_json::Value::Null
    });
    (StatusCode::OK, Json(value))
}

fn json_err(status: StatusCode, msg: impl Into<String>) -> ApiResult {
    let msg = msg.into();
    if status.is_server_error() {
        log::error!("http {status}: {msg}");
    } else {
        log::warn!("http rejected {status}: {msg}");
    }
    (status, Json(serde_json::json!({ "error": msg })))
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

/// Recent log lines for the web UI Logs panel.
async fn logs() -> ApiResult {
    json_ok(&crate::logging::entries())
}

#[derive(Deserialize)]
struct StreamParams {
    path: String,
}

/// Serve a file with Range support (206) for the browser `<video>`; unsupported
/// codecs fall back to FFmpeg frames.
async fn serve_file(path: std::path::PathBuf, req: Request<Body>) -> Response<Body> {
    match tower_http::services::ServeFile::new(path).oneshot(req).await {
        Ok(resp) => resp.map(axum::body::Body::new),
        Err(_) => not_found(),
    }
}

async fn stream(Query(p): Query<StreamParams>, req: Request<Body>) -> Response<Body> {
    let path = std::path::Path::new(&p.path);
    if !media_path(path) {
        return not_found();
    }
    serve_file(path.to_path_buf(), req).await
}

fn audio_cache_dir() -> std::path::PathBuf {
    crate::core::data_dir().join("audio-cache")
}

/// Keep the newest ~20 cached tracks so the cache can't grow unbounded.
fn prune_audio_cache(dir: &std::path::Path) {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "ogg").unwrap_or(false))
        .collect();
    files.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    for p in files.iter().take(files.len().saturating_sub(20)) {
        let _ = std::fs::remove_file(p);
    }
}

/// Transcode the source audio to a cached Vorbis/Ogg track (Chrome rejects
/// this build's audio-only AAC MP4; libvorbis is LGPL-safe).
fn transcode_audio(input: &str) -> Result<std::path::PathBuf, String> {
    if !media_path(std::path::Path::new(input)) {
        return Err("not a media file".into());
    }
    let dir = audio_cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let out = dir.join(format!("{}.ogg", senmei_media::sha256_hex_str(input)));
    if out.is_file() {
        return Ok(out);
    }
    let ff = crate::core::ffmpeg();
    let status = std::process::Command::new(ff)
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-vn", "-c:a", "libvorbis", "-b:a", "96k", "-f", "ogg"])
        .arg(&out)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() || !out.is_file() {
        return Err("audio transcode failed".into());
    }
    prune_audio_cache(&dir);
    Ok(out)
}

/// Serve the source audio as a playable Vorbis/Ogg track (Range support).
async fn audio(Query(p): Query<StreamParams>, req: Request<Body>) -> Response<Body> {
    let input = p.path;
    let out = match tokio::task::spawn_blocking(move || transcode_audio(&input)).await {
        Ok(Ok(o)) => o,
        _ => {
            return json_err(StatusCode::BAD_REQUEST, "audio transcode failed").into_response();
        }
    };
    serve_file(out, req).await
}

/// Empty the buffered log history (Logs panel "Clear").
async fn logs_clear() -> ApiResult {
    crate::logging::clear();
    json_ok(&serde_json::json!({ "ok": true }))
}

async fn backend_info() -> ApiResult {
    json_ok(&senmei_ml::backend_info())
}

async fn probe(Json(p): Json<ProbeParams>) -> ApiResult {
    if !media_path(std::path::Path::new(&p.input)) {
        return json_err(StatusCode::BAD_REQUEST, "not a media file");
    }
    match core::probe_video(&p.input) {
        Ok(info) => json_ok(&info),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

// Same worker as Tauri — scrubbing doesn't respawn ffmpeg per frame.
fn preview_worker() -> &'static senmei_media::PreviewWorker {
    static W: OnceLock<senmei_media::PreviewWorker> = OnceLock::new();
    W.get_or_init(|| senmei_media::PreviewWorker::new(core::ffmpeg()))
}

/// One raw RGB24 frame as the response body; width/height ride in headers so
/// the payload stays binary (ArrayBuffer on the JS side, like the Tauri path).
async fn frame(Json(p): Json<FrameParams>) -> Result<Response<Body>, ApiResult> {
    if !media_path(std::path::Path::new(&p.input)) {
        return Err(json_err(StatusCode::BAD_REQUEST, "not a media file"));
    }
    match preview_worker().frame(&p.input, p.position_ms) {
        Ok(f) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header("x-frame-width", f.width.to_string())
            .header("x-frame-height", f.height.to_string())
            .body(Body::from(f.data))
            .expect("frame response")),
        Err(e) => Err(json_err(StatusCode::BAD_REQUEST, e)),
    }
}

async fn compare(Json(p): Json<CompareParams>) -> ApiResult {
    match core::compare_sample(&p.original, &p.rendered) {
        Ok(v) => json_ok(&v),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn scan_folder(Json(p): Json<ScanParams>) -> ApiResult {
    match core::scan_folder(&p.dir) {
        Ok(files) => json_ok(&files),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn download_model(Json(p): Json<DownloadParams>) -> ApiResult {
    #[cfg(feature = "render")]
    {
        match tokio::task::spawn_blocking(move || {
            core::download_model(&p.model_id, |_, _| {})
        })
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

async fn render_start(Json(cfg): Json<core::RenderConfig>) -> ApiResult {
    #[cfg(feature = "render")]
    {
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

async fn render_status() -> ApiResult {
    #[cfg(feature = "render")]
    {
        json_ok(&core::render_status())
    }
    #[cfg(not(feature = "render"))]
    json_err(StatusCode::SERVICE_UNAVAILABLE, "render not compiled in")
}

async fn render_cancel() -> ApiResult {
    #[cfg(feature = "render")]
    {
        core::cancel_render();
        json_ok(&serde_json::json!({ "cancelled": true }))
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
        .route("/api/logs", get(logs))
        .route("/api/logs/clear", post(logs_clear))
        .route("/api/stream", get(stream))
        .route("/api/audio", get(audio))
        .route("/api/probe", post(probe))
        .route("/api/frame", post(frame))
        .route("/api/compare", post(compare))
        .route("/api/scan-folder", post(scan_folder))
        .route("/api/download-model", post(download_model))
        .route("/api/render", post(render_start))
        .route("/api/render/status", get(render_status))
        .route("/api/render/cancel", post(render_cancel));

    // The built UI is same-origin, so CORS is only needed for the Vite dev
    // server (localhost:1420 → :8765). Locking origins/methods keeps a random
    // website from reading localhost responses (arbitrary file access).
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin([
            "http://localhost:1420".parse().unwrap(),
            "http://127.0.0.1:1420".parse().unwrap(),
        ])
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE])
        .expose_headers([
            header::CONTENT_TYPE,
            "x-frame-width".parse().unwrap(),
            "x-frame-height".parse().unwrap(),
        ]);

    match web_dir {
        Some(dir) => {
            let serve =
                tower_http::services::ServeDir::new(dir).append_index_html_on_directories(true);
            api.layer(cors).fallback_service(serve)
        }
        // No dir given: serve the UI embedded in the binary (SPA fallback).
        None => api.layer(cors).fallback(embedded_fallback),
    }
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::router;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app() -> axum::Router {
        // No web dir: embedded UI (empty in bare test builds) + SPA fallback.
        router(None)
    }

    async fn send(req: Request<Body>) -> (StatusCode, Vec<u8>) {
        let resp = app().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec();
        (status, bytes)
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    fn post_json(uri: &str, json: &str) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(json.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let (status, body) = send(get("/api/health")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(String::from_utf8(body).unwrap(), "ok");
    }

    #[tokio::test]
    async fn backend_info_is_camel_case_json() {
        let (status, body) = send(get("/api/backend-info")).await;
        assert_eq!(status, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v.get("vulkanCompiled").is_some(),
            "missing vulkanCompiled: {v}"
        );
        assert!(v.get("libtorchCompiled").is_some());
    }

    #[tokio::test]
    async fn settings_schema_returns_object() {
        let (status, body) = send(get("/api/settings-schema")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(serde_json::from_slice::<serde_json::Value>(&body)
            .unwrap()
            .is_object());
    }

    #[tokio::test]
    async fn models_returns_array() {
        let (status, body) = send(get("/api/models")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(serde_json::from_slice::<serde_json::Value>(&body)
            .unwrap()
            .is_array());
    }

    #[tokio::test]
    async fn unknown_api_path_404s_instead_of_spa() {
        let (status, _) = send(get("/api/does-not-exist")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn probe_missing_file_returns_400() {
        let (status, body) = send(post_json(
            "/api/probe",
            r#"{"input":"/nonexistent/video.mkv"}"#,
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(serde_json::from_slice::<serde_json::Value>(&body)
            .unwrap()
            .get("error")
            .is_some());
    }

    #[tokio::test]
    async fn scan_folder_missing_dir_returns_400() {
        let (status, _) = send(post_json(
            "/api/scan-folder",
            r#"{"dir":"/nonexistent/dir"}"#,
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn cors_only_allows_known_origins() {
        // No Origin header → not a CORS request, no allow-origin header.
        let resp = app().oneshot(get("/api/health")).await.unwrap();
        assert!(resp.headers().get("access-control-allow-origin").is_none());

        // Unknown cross-origin site → browser blocks reading the response.
        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/health")
            .header("origin", "https://evil.example")
            .body(Body::empty())
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        assert!(resp.headers().get("access-control-allow-origin").is_none());

        // Vite dev origin → allow-origin echoes the known origin.
        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/health")
            .header("origin", "http://localhost:1420")
            .body(Body::empty())
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .map(|v| v.to_str().unwrap()),
            Some("http://localhost:1420")
        );
    }

    #[tokio::test]
    async fn get_on_post_endpoint_is_405() {
        let (status, _) = send(get("/api/probe")).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    }
}
