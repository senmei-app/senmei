//! HTTP adapter tests (router end-to-end).

use super::*;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use std::path::Path;
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
async fn suggest_missing_file_returns_400() {
    let (status, body) = send(post_json(
        "/api/suggest",
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
async fn thumbnail_missing_file_returns_400() {
    let (status, body) = send(post_json(
        "/api/thumbnail",
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

async fn send_with(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
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

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("senmei-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn allowed_roots_cover_canonical_children_only() {
    let root = tmpdir("roots");
    let sub = root.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let f = sub.join("a.mp4");
    std::fs::write(&f, b"x").unwrap();

    let state = AppState::default();
    register_root(&state, &root);
    assert!(is_allowed(&state, &f));
    // Sibling outside the root stays blocked.
    let outside = std::env::temp_dir().join(format!("senmei-outside-{}", std::process::id()));
    std::fs::write(&outside, b"x").unwrap();
    assert!(!is_allowed(&state, &outside));
    // `..` traversal is rejected even under the root.
    let trav = format!("{}/../sub/../a.mp4", root.display());
    assert!(!is_allowed(&state, Path::new(&trav)));

    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_file(&outside).ok();
}

#[test]
fn register_parent_covers_whole_directory() {
    let root = tmpdir("parent");
    let a = root.join("a.mp4");
    let b = root.join("b.mp4");
    std::fs::write(&a, b"x").unwrap();
    std::fs::write(&b, b"x").unwrap();

    let state = AppState::default();
    register_root(&state, a.parent().unwrap());
    assert!(is_allowed(&state, &a));
    assert!(is_allowed(&state, &b));

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn cross_site_scan_folder_is_blocked() {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/scan-folder")
        .header("content-type", "application/json")
        .header("sec-fetch-site", "cross-site")
        .body(Body::from(r#"{"dir":"/tmp"}"#))
        .unwrap();
    let (status, _) = send(req).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn cross_site_with_dev_origin_is_allowed() {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/scan-folder")
        .header("content-type", "application/json")
        .header("sec-fetch-site", "cross-site")
        .header("origin", "http://localhost:1420")
        .body(Body::from(r#"{"dir":"/nonexistent-xyz"}"#))
        .unwrap();
    // Gate passes (dev origin) → falls through to the not-a-directory 400.
    let (status, _) = send(req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn no_sec_fetch_site_is_not_gated() {
    // curl/agents send no Sec-Fetch-Site → not blocked, falls to the 400.
    let (status, _) = send(post_json(
        "/api/scan-folder",
        r#"{"dir":"/nonexistent-xyz"}"#,
    ))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn stream_rejects_unopened_and_serves_opened() {
    let root = tmpdir("stream");
    let f = root.join("clip.mp4");
    std::fs::write(&f, b"data").unwrap();
    let uri = format!("/api/stream?path={}", f.to_string_lossy());

    // Not registered → blocked by the allowed-roots check.
    let (status, _) = send(get(&uri)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Parent registered (e.g. via probe/scan) → served.
    let state = AppState::default();
    register_root(&state, &root);
    let (status, _) = send_with(router_with_state(None, state), get(&uri)).await;
    assert_eq!(status, StatusCode::OK);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn media_path_case_insensitive() {
    let dir = tmpdir("media_case");
    // Create real files so is_file() passes.
    for name in &["a.mp4", "b.MP4", "c.Mp4", "d.FLAC", "e.Flac", "f.txt", "g.PNG", "h.sh"] {
        std::fs::write(dir.join(name), b"x").unwrap();
    }
    assert!(media_path(&dir.join("a.mp4")));
    assert!(media_path(&dir.join("b.MP4")));
    assert!(media_path(&dir.join("c.Mp4")));
    assert!(media_path(&dir.join("d.FLAC")));
    assert!(media_path(&dir.join("e.Flac")));
    assert!(!media_path(&dir.join("f.txt")));
    assert!(!media_path(&dir.join("g.PNG"))); // PNG not in list
    assert!(!media_path(&dir.join("h.sh")));
    std::fs::remove_dir_all(&dir).ok();
}
