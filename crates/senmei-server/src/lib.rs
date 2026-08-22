//! Headless Senmei service: transport-agnostic `senmei-core` + adapters.
//!
//! Decision (2026-08-19): `senmei-core` = thin service (probe/models/
//! render/queue + license & confirm gates) with adapters. **MCP (stdio) first**;
//! REST/HTTP is an optional cargo feature, added only when a real consumer
//! exists (YAGNI). MCP is a transport, not the core — an HTTP API later must
//! not require a refactor.

pub use senmei_core::core;
pub mod logging;
pub mod mcp;
#[cfg(feature = "http")]
pub mod http;

use rmcp::{ServiceExt, transport::stdio};

/// Run the headless service (HTTP or MCP over stdio) from the `senmei`
/// binary. `mcp` wins when both flags are set.
pub async fn run_headless(http_port: u16, mcp: bool) -> anyhow::Result<()> {
    logging::init(&core::data_dir());
    if mcp {
        log::info!("senmei: serving MCP over stdio");
        let service = mcp::SenmeiServer.serve(stdio()).await?;
        service.waiting().await?;
        return Ok(());
    }
    #[cfg(feature = "http")]
    {
        log::info!("senmei: HTTP on http://127.0.0.1:{http_port}");
        let app = http::router(None);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", http_port)).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
    #[cfg(not(feature = "http"))]
    {
        let _ = http_port;
        anyhow::bail!("`http` feature not enabled (build with --features http)")
    }
}
