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
const DEBLUR_MODEL: &str = "nafnet-gopro-width32";

/// Unified inputs for determining suggested pipeline steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuggestInput {
    pub anime: bool,
    pub blurry: bool,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

/// Pure rule engine that maps video metadata and classification to pipeline steps.
pub fn build_suggested_steps(input: &SuggestInput) -> Vec<serde_json::Value> {
    let max_dim = input.width.max(input.height) as u64;
    let fps = input.fps;

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
        let model_id = if input.anime {
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
    // Live footage is often noisy or blurry.
    if !input.anime {
        if input.blurry {
            steps.push(serde_json::json!({
                "stepType": "deblur",
                "params": { "amount": 0.5, "modelId": DEBLUR_MODEL }
            }));
        } else {
            steps.push(serde_json::json!({
                "stepType": "denoise",
                "params": { "radius": 1, "modelId": DENOISE_MODEL }
            }));
        }
    }
    steps.push(serde_json::json!({ "stepType": "output", "params": {} }));
    steps
}

/// Probe `input` and suggest a default pipeline (content-aware defaults): anime
/// vs live-action, input resolution, frame rate. Returns a JSON string
/// (`{ anime, steps: [{ stepType, params }] }`).
pub fn suggest_pipeline(input: &str) -> Result<String, String> {
    let info = probe_video(input)?;
    let ffmpeg_path = ffmpeg();
    let duration_ms = (info.duration * 1000.0) as u64;

    let anime = senmei_media::is_anime(
        &ffmpeg_path,
        std::path::Path::new(input),
        duration_ms,
    );

    let blurry = if !anime {
        senmei_media::is_blurry(
            &ffmpeg_path,
            std::path::Path::new(input),
            duration_ms,
        )
    } else {
        false
    };

    let suggest_input = SuggestInput {
        anime,
        blurry,
        width: info.width,
        height: info.height,
        fps: info.fps,
    };

    let steps = build_suggested_steps(&suggest_input);

    log::info!(
        "suggest_pipeline: {} {}x{} {fps:.0}fps anime={anime} blurry={blurry}",
        input,
        info.width,
        info.height,
        fps = info.fps
    );
    serde_json::to_string(&serde_json::json!({ "anime": anime, "steps": steps }))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_rules() {
        struct TestCase {
            name: &'static str,
            input: SuggestInput,
            expected_steps: Vec<(&'static str, serde_json::Value)>,
        }

        let cases = vec![
            TestCase {
                name: "low-fps low-res anime",
                input: SuggestInput {
                    anime: true,
                    blurry: false,
                    width: 640,
                    height: 360,
                    fps: 23.976,
                },
                expected_steps: vec![
                    ("interpolation", serde_json::json!({ "fpsMultiplier": 2, "modelId": "rife-v4.6" })),
                    ("upscale", serde_json::json!({ "scale": 4, "modelId": "realesrgan-animevideo-x4" })),
                    ("output", serde_json::json!({})),
                ],
            },
            TestCase {
                name: "high-fps high-res live action (clean/noisy)",
                input: SuggestInput {
                    anime: false,
                    blurry: false,
                    width: 3840,
                    height: 2160,
                    fps: 60.0,
                },
                expected_steps: vec![
                    ("denoise", serde_json::json!({ "radius": 1, "modelId": "drunet-color" })),
                    ("output", serde_json::json!({})),
                ],
            },
            TestCase {
                name: "blurry live action requiring deblur",
                input: SuggestInput {
                    anime: false,
                    blurry: true,
                    width: 1280,
                    height: 720,
                    fps: 30.0,
                },
                expected_steps: vec![
                    ("upscale", serde_json::json!({ "scale": 4, "modelId": "bsrgan" })),
                    ("deblur", serde_json::json!({ "amount": 0.5, "modelId": "nafnet-gopro-width32" })),
                    ("output", serde_json::json!({})),
                ],
            },
            TestCase {
                name: "borderline fps threshold (exactly 30fps) - no interpolation",
                input: SuggestInput {
                    anime: true,
                    blurry: false,
                    width: 1920,
                    height: 1080,
                    fps: 30.0,
                },
                expected_steps: vec![
                    ("upscale", serde_json::json!({ "scale": 2, "modelId": "realesrgan-animevideo-x2" })),
                    ("output", serde_json::json!({})),
                ],
            },
        ];

        for case in cases {
            let steps = build_suggested_steps(&case.input);
            assert_eq!(
                steps.len(),
                case.expected_steps.len(),
                "Case '{}' failed on steps count",
                case.name
            );

            for (i, step) in steps.iter().enumerate() {
                let (expected_type, expected_params) = &case.expected_steps[i];
                assert_eq!(
                    step.get("stepType").and_then(|v| v.as_str()),
                    Some(*expected_type),
                    "Case '{}' step {} type mismatch",
                    case.name,
                    i
                );
                assert_eq!(
                    step.get("params"),
                    Some(expected_params),
                    "Case '{}' step {} params mismatch",
                    case.name,
                    i
                );
            }
        }
    }
}
