//! Headless Senmei service binary. Default: MCP over stdio; `--http` (or
//! `SENMEI_HTTP`) serves the full web UI + REST API instead (headless, no
//! display server needed).

use rmcp::{ServiceExt, transport::stdio};
use senmei_server::mcp::SenmeiServer;

#[cfg(feature = "http")]
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    #[cfg(feature = "http")]
    if std::env::var("SENMEI_HTTP").is_ok() || std::env::args().any(|a| a == "--http") {
        return serve_http().await;
    }

    log::info!("senmei-server: serving MCP over stdio");
    let service = SenmeiServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Serve the full web UI + REST API (headless; no display server needed).
#[cfg(feature = "http")]
async fn serve_http() -> anyhow::Result<()> {
    let port: u16 = std::env::var("SENMEI_HTTP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let web_dir = web_dir();
    log::info!("senmei-server: HTTP on http://127.0.0.1:{port}");
    let app = senmei_server::http::router(web_dir);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Resolve the built web UI dir: `SENMEI_WEB_DIR` env, else the repo checkout.
#[cfg(feature = "http")]
fn web_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SENMEI_WEB_DIR") {
        return Some(PathBuf::from(p));
    }
    let anchored = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/app/dist");
    anchored.is_dir().then_some(anchored)
}
