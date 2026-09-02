//! HTTP adapter over the core service — serves the full web UI + REST API.
//! Same license/confirm gates as MCP (they live in `core`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, Method, Request, Response, StatusCode},
    middleware::{self, Next},
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

/// Dirs the local user has opened/scanned (canonicalized). The HTTP service
/// only serves/processes media under one of these roots, so a random website
/// can't reach arbitrary files — even via a `<video>` src (which bypasses CORS).
#[derive(Clone, Default)]
struct AppState {
    roots: Arc<Mutex<HashSet<PathBuf>>>,
}

/// Vite dev origins — the only legit cross-site callers (UI on :1420, API on
/// :8765). Mirrors the CORS allow-list below.
const DEV_ORIGINS: [&str; 2] = ["http://localhost:1420", "http://127.0.0.1:1420"];

/// Canonicalize a real path; reject obvious `..` traversal first.
fn canonical(p: &Path) -> Option<PathBuf> {
    if p.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return None;
    }
    std::fs::canonicalize(p).ok()
}

/// Canonical path of `p` when it's inside a registered root (else None). The
/// canonical path is what reaches the filesystem sinks — `..`, symlinks, and
/// out-of-root paths are all resolved/rejected before any path use.
fn resolve_allowed(state: &AppState, p: &Path) -> Option<PathBuf> {
    let c = canonical(p)?;
    let roots = state.roots.lock().unwrap();
    roots.iter().find(|r| c.starts_with(r)).map(|_| c)
}

fn is_allowed(state: &AppState, p: &Path) -> bool {
    resolve_allowed(state, p).is_some()
}

fn register_root(state: &AppState, dir: &Path) {
    if let Some(c) = canonical(dir) {
        state.roots.lock().unwrap().insert(c);
    }
}

/// Register the parent dir of a just-opened file (probe/thumbnail/render).
fn register_parent(state: &AppState, p: &Path) {
    if let Some(parent) = p.parent() {
        register_root(state, parent);
    }
}

/// Block browser cross-site requests to the path/side-effect routes, so a
/// random website can't register roots, enumerate folders, or start renders.
/// `/api/stream` + `/api/audio` are excluded — they're `<video>`/`<audio>`
/// sources (no Origin header) and are already confined by the allowed-roots
/// check. Non-browser clients (curl/agents) send no `Sec-Fetch-Site` → allowed.
async fn require_local_client(req: Request<Body>, next: Next) -> Response<Body> {
    let path = req.uri().path();
    if path.starts_with("/api/stream") || path.starts_with("/api/audio") {
        return next.run(req).await;
    }
    let Some(site) = req.headers().get("sec-fetch-site") else {
        return next.run(req).await; // curl/agents/non-browser
    };
    let site = site.to_str().unwrap_or("");
    if site == "same-origin" || site == "same-site" || site == "none" {
        return next.run(req).await;
    }
    // cross-site: allow only the Vite dev origins.
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok());
    if origin.is_some_and(|o| DEV_ORIGINS.contains(&o)) {
        return next.run(req).await;
    }
    (StatusCode::FORBIDDEN, "blocked cross-site request").into_response()
}

const VIDEO_EXTS: &[&str] = &[
    "mp4", "m4v", "mkv", "webm", "mov", "avi", "ts", "m2ts", "mts", "flv", "wmv", "mpg", "mpeg",
    "vob", "3gp", "f4v", "ogv",
];
const AUDIO_EXTS: &[&str] = &[
    "mp3", "flac", "ogg", "oga", "opus", "wav", "aac", "m4a", "wma", "ac3", "ape",
];

/// The localhost REST API serves caller-supplied paths (`stream`/`audio`/
/// `frame`/`probe`). Restrict to real media files so a cross-origin page can't
/// read arbitrary local files (CORS is locked down too — see `router`).
fn media_path(p: &std::path::Path) -> bool {
    if !p.is_file() {
        return false;
    }
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            VIDEO_EXTS
                .iter()
                .chain(AUDIO_EXTS)
                .any(|&m| m.eq_ignore_ascii_case(ext))
        })
}

/// Canonicalize + media-check + register parent in one call. Returns `None`
/// (and sends an error response) when the path is invalid or not a media file.
fn resolve_media_input(state: &AppState, input: &str) -> Option<PathBuf> {
    let path = canonical(Path::new(input))?;
    if !media_path(&path) {
        return None;
    }
    register_parent(state, &path);
    Some(path)
}

#[derive(Deserialize)]
struct ProbeParams {
    input: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ThumbnailParams {
    input: String,
    max_w: Option<u32>,
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

#[derive(Deserialize)]
struct SuggestParams {
    /// Path to a video file to suggest a pipeline for.
    input: String,
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
    match tower_http::services::ServeFile::new(path)
        .oneshot(req)
        .await
    {
        Ok(resp) => resp.map(axum::body::Body::new),
        Err(_) => not_found(),
    }
}

async fn stream(
    State(state): State<AppState>,
    Query(p): Query<StreamParams>,
    req: Request<Body>,
) -> Response<Body> {
    let Some(path) = resolve_allowed(&state, Path::new(&p.path)) else {
        return not_found();
    };
    if !media_path(&path) {
        return not_found();
    }
    serve_file(path, req).await
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
    let status = senmei_media::process::hidden(ff)
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
async fn audio(
    State(state): State<AppState>,
    Query(p): Query<StreamParams>,
    req: Request<Body>,
) -> Response<Body> {
    let Some(input) = resolve_allowed(&state, Path::new(&p.path)) else {
        return not_found();
    };
    let input = input.to_string_lossy().into_owned();
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

async fn probe(State(state): State<AppState>, Json(p): Json<ProbeParams>) -> ApiResult {
    let Some(input) = resolve_media_input(&state, &p.input) else {
        return json_err(StatusCode::BAD_REQUEST, "not a media file");
    };
    match core::probe_video(&input.to_string_lossy()) {
        Ok(info) => json_ok(&info),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

/// Content-aware default pipeline suggestion — same payload as the Tauri
/// command (returns the `{ anime, steps }` object as JSON).
async fn suggest(State(state): State<AppState>, Json(p): Json<SuggestParams>) -> ApiResult {
    let Some(input) = resolve_media_input(&state, &p.input) else {
        return json_err(StatusCode::BAD_REQUEST, "not a media file");
    };
    let input = input.to_string_lossy().into_owned();
    let out = tokio::task::spawn_blocking(move || core::suggest_pipeline(&input)).await;
    match out {
        Ok(Ok(json)) => match serde_json::from_str::<serde_json::Value>(&json) {
            Ok(v) => json_ok(&v),
            Err(e) => json_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("suggest produced invalid JSON: {e}"),
            ),
        },
        Ok(Err(e)) => json_err(StatusCode::BAD_REQUEST, e),
        Err(e) => json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("suggest task failed: {e}"),
        ),
    }
}

/// Small JPEG thumbnail as a `data:` URL (same payload as the Tauri IPC).
async fn thumbnail(State(state): State<AppState>, Json(p): Json<ThumbnailParams>) -> ApiResult {
    let Some(input) = resolve_media_input(&state, &p.input) else {
        return json_err(StatusCode::BAD_REQUEST, "not a media file");
    };
    match core::thumbnail(&input.to_string_lossy(), p.max_w.unwrap_or(160)) {
        Ok((data_url, info)) => json_ok(&serde_json::json!({ "data": data_url, "info": info })),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

// Same worker as Tauri — scrubbing doesn't respawn ffmpeg per frame.
fn preview_worker() -> &'static senmei_media::PreviewWorker {
    static W: OnceLock<senmei_media::PreviewWorker> = OnceLock::new();
    W.get_or_init(|| senmei_media::PreviewWorker::new(crate::core::data_dir()))
}

/// One raw RGB24 frame as the response body; width/height ride in headers so
/// the payload stays binary (ArrayBuffer on the JS side, like the Tauri path).
async fn frame(
    State(state): State<AppState>,
    Json(p): Json<FrameParams>,
) -> Result<Response<Body>, ApiResult> {
    let Some(input) = resolve_media_input(&state, &p.input) else {
        return Err(json_err(
            StatusCode::BAD_REQUEST,
            "not an opened media file",
        ));
    };
    match preview_worker().frame(&input.to_string_lossy(), p.position_ms) {
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

async fn compare(State(state): State<AppState>, Json(p): Json<CompareParams>) -> ApiResult {
    if !is_allowed(&state, Path::new(&p.original)) || !is_allowed(&state, Path::new(&p.rendered)) {
        return json_err(StatusCode::BAD_REQUEST, "not an opened media file");
    }
    match core::compare_sample(&p.original, &p.rendered) {
        Ok(v) => json_ok(&v),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn scan_folder(State(state): State<AppState>, Json(p): Json<ScanParams>) -> ApiResult {
    let dir = Path::new(&p.dir);
    if !dir.is_dir() {
        return json_err(StatusCode::BAD_REQUEST, "not a directory");
    }
    register_root(&state, dir);
    match core::scan_folder(&p.dir) {
        Ok(files) => json_ok(&files),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

/// Build the HTTP router: REST API + optional static UI (ServeDir fallback).
pub fn router(web_dir: Option<PathBuf>) -> Router {
    router_with_state(web_dir, AppState::default())
}

/// Test seam: seed the allowed-roots state directly.
fn router_with_state(web_dir: Option<PathBuf>, state: AppState) -> Router {
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
        .route("/api/suggest", post(suggest))
        .route("/api/thumbnail", post(thumbnail))
        .route("/api/frame", post(frame))
        .route("/api/compare", post(compare))
        .route("/api/scan-folder", post(scan_folder))
        .route("/api/download-model", post(download_model))
        .route("/api/render", post(render_start))
        .route("/api/render/status", get(render_status))
        .route("/api/render/cancel", post(render_cancel))
        .layer(middleware::from_fn(require_local_client))
        .with_state(state);

    // The built UI is same-origin, so CORS is only needed for the Vite dev
    // server (localhost:1420 → :8765). Locking origins/methods keeps a random
    // website from reading localhost responses (arbitrary file access).
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(
            DEV_ORIGINS
                .iter()
                .map(|o| o.parse().unwrap())
                .collect::<Vec<_>>(),
        )
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

mod render;
#[cfg(all(test, feature = "http"))]
mod tests;

use render::{download_model, render_cancel, render_start, render_status};
