//! MCP (stdio) adapter over the core service.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_router,
};
use serde::Deserialize;

use crate::core;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ProbeParams {
    /// Path to a video file.
    input: String,
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
