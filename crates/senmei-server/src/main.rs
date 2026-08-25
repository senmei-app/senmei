//! Headless Senmei service binary. Default: MCP over stdio; `--server` (or
//! `SENMEI_HTTP`) serves the full web UI + REST API instead (headless, no
//! display server needed).

use clap::Parser;
use rmcp::{transport::stdio, ServiceExt};
use senmei_server::mcp::SenmeiServer;

#[cfg(feature = "http")]
use std::path::PathBuf;

/// Headless Senmei service.
#[derive(Parser, Debug)]
#[command(
    name = "senmei-server",
    version,
    about = "Headless Senmei service (MCP over stdio by default)"
)]
struct Cli {
    /// Serve the full web UI + REST API over HTTP.
    #[arg(short, long, alias = "http")]
    server: bool,

    /// HTTP listen port (default 8765, or $SENMEI_HTTP_PORT).
    #[arg(short = 'p', long, env = "SENMEI_HTTP_PORT", default_value_t = 8765)]
    http_port: u16,

    /// Serve MCP over stdio (the default).
    #[arg(short, long)]
    mcp_server: bool,

    /// Built web UI dir (default: repo checkout, or $SENMEI_WEB_DIR).
    #[arg(long, env = "SENMEI_WEB_DIR")]
    #[cfg(feature = "http")]
    web_dir: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // burn autotune stack-overflows on RADV with the default 2 MiB (see the
    // GUI main); set before the runtime spawns any thread so workers and the
    // render threads inherit it.
    std::env::set_var("RUST_MIN_STACK", "33554432");
    senmei_server::logging::init(&senmei_core::core::data_dir());
    let cli = Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run(&cli))
}

async fn run(cli: &Cli) -> anyhow::Result<()> {
    #[cfg(feature = "http")]
    if cli.server || std::env::var("SENMEI_HTTP").is_ok() {
        return serve_http(cli).await;
    }

    #[cfg(not(feature = "http"))]
    if cli.server || std::env::var("SENMEI_HTTP").is_ok() {
        anyhow::bail!("--server needs the `http` feature (build with --features http)");
    }

    log::info!("senmei-server: serving MCP over stdio");
    let service = SenmeiServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Serve the full web UI + REST API (headless; no display server needed).
#[cfg(feature = "http")]
async fn serve_http(cli: &Cli) -> anyhow::Result<()> {
    // clap resolves CLI > $SENMEI_HTTP_PORT > default.
    let port = cli.http_port;
    let web_dir = web_dir(cli.web_dir.clone());
    log::info!("senmei-server: HTTP on http://127.0.0.1:{port}");
    let app = senmei_server::http::router(web_dir);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Resolve the built web UI dir: `--web-dir`/`SENMEI_WEB_DIR` (merged by
/// clap), else the repo checkout.
#[cfg(feature = "http")]
fn web_dir(cli_dir: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(p) = cli_dir {
        return Some(p);
    }
    let anchored = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/app/dist");
    anchored.is_dir().then_some(anchored)
}
