// Release builds run as GUI app (no console window); debug keeps the console for logs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use tauri::Manager;

/// Senmei (鮮明) — video enhancer. GUI by default; `--server`/`--mcp-server`
/// run the same service headless (web UI embedded in the binary).
#[derive(Parser)]
#[command(name = "senmei", version, about = "Senmei (鮮明) — video enhancer")]
struct Cli {
    /// Serve the web UI + REST API over HTTP (headless, no GUI).
    #[arg(short, long)]
    server: bool,

    /// HTTP listen port (default 8765, or $SENMEI_HTTP_PORT).
    #[arg(short = 'p', long, env = "SENMEI_HTTP_PORT", default_value_t = 8765)]
    http_port: u16,

    /// Serve MCP over stdio (headless).
    #[arg(short, long)]
    mcp_server: bool,
}

fn main() -> anyhow::Result<()> {
    // burn autotune stack-overflows on RADV with the default 2 MiB; set before
    // any thread spawns so both the GUI and headless runtimes inherit it.
    std::env::set_var("RUST_MIN_STACK", "33554432");
    let cli = Cli::parse();
    if cli.server || cli.mcp_server {
        return run_headless(cli.http_port, cli.mcp_server);
    }

    senmei_app::log_hub::init();
    log::info!("Senmei starting");

    let builder = senmei_app::specta_builder();

    #[cfg(debug_assertions)]
    builder
        .export(
            specta_typescript::Typescript::default(),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../packages/bridge/src/bindings.ts"
            ),
        )
        .expect("failed to export typescript bindings");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            senmei_app::log_hub::attach(app.handle());
            // Packaged app: make the bundled model catalog reachable in the
            // writable data dir (dev falls back to the repo checkout).
            let resource_dir = app.path().resource_dir().ok();
            if let Err(e) = senmei_app::models::ensure_catalog(resource_dir.as_deref()) {
                log::warn!("model catalog not materialized: {e}");
            }
            Ok(())
        })
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("error while running tauri application: {e}"))?;
    Ok(())
}

/// Headless service. RUST_MIN_STACK is set at the top of `main`; the runtime
/// threads inherit it from there.
fn run_headless(http_port: u16, mcp: bool) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(senmei_server::run_headless(http_port, mcp))
}
