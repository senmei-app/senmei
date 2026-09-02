//! Content-aware default pipeline suggestion (probe + classify). Shared by the
//! Tauri command and the HTTP adapter so headless clients get the same
//! defaults as the GUI.

use super::{ffmpeg, probe_video};

/// Suggestion knobs: interpolation FPS floor, upscale breakpoints and the model
/// IDs the suggestion maps to (mirrors the registry catalog).
const FPS_INTERP_THRESHOLD: f64 = 30.0;
const UPSCALE_1080P: u64 = 1920;
const UPSCALE_4K: u64 = 3840;
const INTERP_MODEL: &str = "rife-v4.6";
const ANIME_UPSCALE_X4: &str = "realesrgan-animevideo-x4";
const ANIME_UPSCALE_X2: &str = "realesrgan-animevideo-x2";
const GENERAL_UPSCALE: &str = "bsrgan";
const DENOISE_MODEL: &str = "drunet-color";

/// Probe `input` and suggest a default pipeline (content-aware defaults): anime
/// vs live-action, input resolution, frame rate. Returns a JSON string
/// (`{ anime, steps: [{ stepType, params }] }`).
pub fn suggest_pipeline(input: &str) -> Result<String, String> {
    let info = probe_video(input)?;
    let anime = senmei_media::is_anime(
        &ffmpeg(),
        std::path::Path::new(input),
        (info.duration * 1000.0) as u64,
    );
    let max_dim = info.width.max(info.height) as u64;
    let fps = info.fps;

    let mut steps = Vec::new();
    // Interpolate low-fps content to 2× (24 → 48 etc.).
    if fps > 0.0 && fps < FPS_INTERP_THRESHOLD {
        steps.push(serde_json::json!({
            "stepType": "interpolation",
            "params": { "fpsMultiplier": 2, "modelId": INTERP_MODEL }
        }));
    }
    // Upscale to at least 1080p, max 4×; anime gets the fast anime upscaler.
    let scale = if max_dim < UPSCALE_1080P {
        4
    } else if max_dim < UPSCALE_4K {
        2
    } else {
        1
    };
    if scale > 1 {
        let model_id = if anime {
            if scale >= 4 {
                ANIME_UPSCALE_X4
            } else {
                ANIME_UPSCALE_X2
            }
        } else {
            GENERAL_UPSCALE
        };
        steps.push(serde_json::json!({
            "stepType": "upscale",
            "params": { "scale": scale, "modelId": model_id }
        }));
    }
    // Live footage is often noisy; suggest a denoise pass (anime is clean).
    if !anime {
        steps.push(serde_json::json!({
            "stepType": "denoise",
            "params": { "radius": 1, "modelId": DENOISE_MODEL }
        }));
    }
    steps.push(serde_json::json!({ "stepType": "output", "params": {} }));

    log::info!(
        "suggest_pipeline: {} {}x{} {fps:.0}fps anime={anime}",
        input,
        info.width,
        info.height
    );
    serde_json::to_string(&serde_json::json!({ "anime": anime, "steps": steps }))
        .map_err(|e| e.to_string())
}
