//! MCP (stdio) adapter over the core service.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_router,
};
use serde::Deserialize;

#[cfg(feature = "render")]
use base64::Engine as _;

use crate::core;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ProbeParams {
    /// Path to a video file.
    input: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CompareParams {
    /// Original (source) video path.
    original: String,
    /// Rendered sample path (from render_sample).
    rendered: String,
}

#[derive(Clone)]
pub struct SenmeiServer;

#[tool_router(server_handler)]
impl SenmeiServer {
    #[tool(description = "Health check: returns ok when the server is up")]
    async fn health(&self) -> String {
        "ok".into()
    }

    #[tool(description = "Probe a video file: dims, fps, duration, rotation, HDR")]
    async fn probe_video(
        &self,
        Parameters(args): Parameters<ProbeParams>,
    ) -> Result<CallToolResult, McpError> {
        let result = core::probe_video(&args.input);
        match result {
            Ok(info) => json_ok(&info),
            Err(e) => json_err(format!("probe failed: {e}")),
        }
    }

    #[tool(description = "List the model catalog (id, kind, scale, loadable, license)")]
    async fn list_models(&self) -> Result<CallToolResult, McpError> {
        let models: Vec<serde_json::Value> = core::list_models()
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "kind": m.kind,
                    "scale": m.scale,
                    "loadable": m.loadable,
                    "license": m.license,
                    "licenseBlocked": m.license_blocked(),
                })
            })
            .collect();
        json_ok(&models)
    }

    #[tool(description = "FFmpeg status: resolved binary path + capabilities")]
    async fn get_ffmpeg_status(&self) -> Result<CallToolResult, McpError> {
        json_ok(&core::ffmpeg_status())
    }

    #[tool(description = "Settings schema: render-config JSON Schema + model slots (which models fill which field) + hard constraints")]
    async fn get_settings_schema(&self) -> Result<CallToolResult, McpError> {
        json_ok(&core::settings_schema())
    }

    #[tool(description = "Render a short sample clip (range render, no confirm gate); returns output path + before/after frames as images")]
    async fn render_sample(
        &self,
        Parameters(args): Parameters<core::RenderConfig>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            return match tokio::task::spawn_blocking(move || core::render_sample(args)).await {
                Ok(Ok(v)) => render_sample_result(v),
                Ok(Err(e)) => json_err(e),
                Err(e) => json_err(format!("render_sample join failed: {e}")),
            };
        }
        #[cfg(not(feature = "render"))]
        {
            let _ = args;
            json_err("render not compiled in (build with --features render)".to_string())
        }
    }

    #[tool(description = "Compare a rendered sample against its original: PSNR (dB) + SSIM on the original resolution")]
    async fn compare_sample(
        &self,
        Parameters(args): Parameters<CompareParams>,
    ) -> Result<CallToolResult, McpError> {
        match core::compare_sample(&args.original, &args.rendered) {
            Ok(metrics) => json_ok(&metrics),
            Err(e) => json_err(e),
        }
    }

    #[tool(description = "Propose a render (validates, does NOT start); confirm with confirm_render")]
    async fn propose_render(
        &self,
        Parameters(args): Parameters<core::RenderConfig>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            return match core::propose_render(args) {
                Ok(msg) => json_ok(&msg),
                Err(e) => json_err(e),
            };
        }
        #[cfg(not(feature = "render"))]
        {
            let _ = args;
            json_err("render not compiled in (build with --features render)".to_string())
        }
    }

    #[tool(description = "Run the previously proposed render (confirmation gate)")]
    async fn confirm_render(&self) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            return match core::confirm_render() {
                Ok(msg) => json_ok(&msg),
                Err(e) => json_err(e),
            };
        }
        #[cfg(not(feature = "render"))]
        json_err("render not compiled in (build with --features render)".to_string())
    }

    #[tool(description = "Cancel the active render")]
    async fn cancel_render(&self) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            core::cancel_render();
            return json_ok(&"ok");
        }
        #[cfg(not(feature = "render"))]
        json_err("render not compiled in (build with --features render)".to_string())
    }

    #[tool(description = "Poll render status (idle/running/done/failed + frame counts)")]
    async fn get_render_status(&self) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            return json_ok(&core::render_status());
        }
        #[cfg(not(feature = "render"))]
        json_err("render not compiled in (build with --features render)".to_string())
    }
}

fn json_ok<T: serde::Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let json = serde_json::to_string(value)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
}

fn json_err(message: String) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

/// Attach a PNG file as an MCP image content block (base64) — lets multimodal
/// clients show the before/after frames directly.
#[cfg(feature = "render")]
fn image_block(path: &str) -> Option<ContentBlock> {
    let bytes = std::fs::read(path).ok()?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(ContentBlock::image(b64, "image/png"))
}

/// `render_sample` result: text summary (paths) + before/after image blocks.
#[cfg(feature = "render")]
fn render_sample_result(v: serde_json::Value) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string(&v)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    let mut blocks = vec![ContentBlock::text(text)];
    for key in ["beforeFrame", "afterFrame"] {
        if let Some(path) = v.get(key).and_then(serde_json::Value::as_str) {
            if let Some(b) = image_block(path) {
                blocks.push(b);
            }
        }
    }
    Ok(CallToolResult::success(blocks))
}
