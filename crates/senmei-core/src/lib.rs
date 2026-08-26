//! Transport-agnostic Senmei core: probe/render/models/queue + license &
//! confirm gates. No Tauri, no webview, no transport — every adapter (MCP,
//! HTTP, GUI) calls into here so the gates live once.

pub mod core;
pub mod logging;
