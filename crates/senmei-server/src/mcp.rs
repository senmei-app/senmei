//! MCP (stdio) adapter over the core service.

use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_router, ErrorData as McpError,
};
use serde::Deserialize;

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
struct DownloadModelParams {
    /// Model id to download weights for.
    model_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ThumbnailParams {
    /// Path to a media file.
    input: String,
    /// Max width for the thumbnail (default 160).
    #[serde(default)]
    max_w: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ScanFolderParams {
    /// Directory path to scan recursively for video files.
    dir: String,
}

/// Simplified params for render_sample — keeps the MCP tool schema small.
/// Fields map 1:1 to `core::RenderConfig`; missing optional fields get defaults.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
struct RenderSampleParams {
    /// Input video path.
    input: String,
    /// Output video path (container guessed from extension).
    output: String,
    /// Upscale model id (kind=upscale); empty = reference upscale only.
    #[serde(default)]
    model_id: Option<String>,
    /// Integer upscale factor (1..=4).
    #[serde(default)]
    scale: Option<u32>,
    /// Render start in ms (for sample clips).
    #[serde(default)]
    start_ms: Option<u64>,
    /// Render end in ms (for sample clips).
    #[serde(default)]
    end_ms: Option<u64>,
    /// Frame-rate multiplier for interpolation (1..=16).
    #[serde(default)]
    fps_multiplier: Option<u32>,
    /// Interpolation model id (kind=interpolate).
    #[serde(default)]
    interp_model: Option<String>,
    /// Denoise model id (kind=denoise).
    #[serde(default)]
    denoise_model_id: Option<String>,
    /// Deblur model id (kind=deblur).
    #[serde(default)]
    deblur_model_id: Option<String>,
}

impl RenderSampleParams {
    fn into_render_config(self) -> core::RenderConfig {
        let filter = if self.denoise_model_id.is_some() || self.deblur_model_id.is_some() {
            Some(core::FilterConfig {
                denoise_model_id: self.denoise_model_id,
                deblur_model_id: self.deblur_model_id,
                ..Default::default()
            })
        } else {
            None
        };
        core::RenderConfig {
            input: self.input,
            output: self.output,
            model_id: self.model_id,
            scale: self.scale,
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            fps_multiplier: self.fps_multiplier,
            interp_model: self.interp_model,
            filter,
            ..Default::default()
        }
    }
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

    #[tool(
        description = "Settings schema: render-config JSON Schema + model slots (which models fill which field) + hard constraints"
    )]
    async fn get_settings_schema(&self) -> Result<CallToolResult, McpError> {
        json_ok(&core::settings_schema())
    }

    #[tool(
        description = "Render a short sample clip (range render, no confirm gate); returns output path + before/after frames as images"
    )]
    async fn render_sample(
        &self,
        Parameters(args): Parameters<RenderSampleParams>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            let config = args.into_render_config();
            return match tokio::task::spawn_blocking(move || core::render_sample(config)).await {
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

    #[tool(
        description = "Compare a rendered sample against its original: PSNR (dB) + SSIM on the original resolution"
    )]
    async fn compare_sample(
        &self,
        Parameters(args): Parameters<CompareParams>,
    ) -> Result<CallToolResult, McpError> {
        match core::compare_sample(&args.original, &args.rendered) {
            Ok(metrics) => json_ok(&metrics),
            Err(e) => json_err(e),
        }
    }

    #[tool(
        description = "Propose a render (validates, does NOT start); confirm with confirm_render"
    )]
    async fn propose_render(
        &self,
        Parameters(args): Parameters<RenderSampleParams>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            let config = args.into_render_config();
            return match core::propose_render(config) {
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

    #[tool(
        description = "Download a model's weights and convert to burnpack (render feature required)"
    )]
    async fn download_model(
        &self,
        Parameters(args): Parameters<DownloadModelParams>,
    ) -> Result<CallToolResult, McpError> {
        #[cfg(feature = "render")]
        {
            return match tokio::task::spawn_blocking(move || {
                core::download_model(&args.model_id, |_, _| {})
            })
            .await
            {
                Ok(Ok(path)) => json_ok(&serde_json::json!({ "bpk": path })),
                Ok(Err(e)) => json_err(e),
                Err(e) => json_err(format!("download_model join failed: {e}")),
            };
        }
        #[cfg(not(feature = "render"))]
        {
            let _ = args;
            json_err("render not compiled in (build with --features render)".to_string())
        }
    }

    #[tool(
        description = "GPU/backend info: Vulkan compiled, libtorch compiled, CUDA available, device count"
    )]
    async fn backend_info(&self) -> Result<CallToolResult, McpError> {
        json_ok(&senmei_ml::backend_info())
    }

    #[tool(description = "Recursively scan a directory for video files")]
    async fn scan_folder(
        &self,
        Parameters(args): Parameters<ScanFolderParams>,
    ) -> Result<CallToolResult, McpError> {
        match core::scan_folder(&args.dir) {
            Ok(files) => json_ok(&files),
            Err(e) => json_err(e),
        }
    }

    #[tool(description = "Small JPEG thumbnail of a media file as base64 (data URL) + probe info")]
    async fn thumbnail(
        &self,
        Parameters(args): Parameters<ThumbnailParams>,
    ) -> Result<CallToolResult, McpError> {
        match core::thumbnail(&args.input, args.max_w.unwrap_or(160)) {
            Ok((data_url, info)) => json_ok(&serde_json::json!({ "data": data_url, "info": info })),
            Err(e) => json_err(e),
        }
    }
}

fn json_ok<T: serde::Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let json =
        serde_json::to_string(value).map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
}

fn json_err(message: String) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

/// Attach a PNG file as an MCP image content block (base64) — lets multimodal
/// clients show the before/after frames directly.
fn image_block(path: &str) -> Option<ContentBlock> {
    let bytes = std::fs::read(path).ok()?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Some(ContentBlock::image(b64, "image/png"))
}

/// `render_sample` result: text summary (paths) + before/after image blocks.
fn render_sample_result(v: serde_json::Value) -> Result<CallToolResult, McpError> {
    let text =
        serde_json::to_string(&v).map_err(|e| McpError::internal_error(e.to_string(), None))?;
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
#[cfg(test)]
mod tests {
    use super::*;

    fn params(input: &str, output: &str) -> RenderSampleParams {
        RenderSampleParams {
            input: input.into(),
            output: output.into(),
            model_id: Some("fallin-soft".into()),
            scale: Some(2),
            start_ms: Some(1000),
            end_ms: Some(2000),
            fps_multiplier: None,
            interp_model: None,
            denoise_model_id: Some("drunet-color".into()),
            deblur_model_id: None,
        }
    }

    #[test]
    fn render_sample_params_map_onto_render_config() {
        let cfg = params("a.mp4", "b.mkv").into_render_config();
        assert_eq!(cfg.input, "a.mp4");
        assert_eq!(cfg.output, "b.mkv");
        assert_eq!(cfg.model_id.as_deref(), Some("fallin-soft"));
        assert_eq!(cfg.scale, Some(2));
        assert_eq!((cfg.start_ms, cfg.end_ms), (Some(1000), Some(2000)));
        let f = cfg.filter.expect("denoise model id sets a filter");
        assert_eq!(f.denoise_model_id.as_deref(), Some("drunet-color"));
        assert_eq!(f.deblur_model_id, None);
    }

    #[test]
    fn no_filters_leave_filter_none() {
        let mut p = params("a.mp4", "b.mkv");
        p.denoise_model_id = None;
        assert!(p.into_render_config().filter.is_none());
    }
}
