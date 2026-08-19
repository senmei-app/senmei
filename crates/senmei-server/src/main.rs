//! Headless Senmei service binary. Default: MCP over stdio.

use rmcp::{ServiceExt, transport::stdio};
use senmei_server::mcp::SenmeiServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    log::info!("senmei-server: serving MCP over stdio");
    let service = SenmeiServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
