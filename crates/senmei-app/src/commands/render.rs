//! Render types and helpers for the Tauri IPC layer.

use senmei_core::core;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct RenderProgress {
    pub frames_processed: u64,
    pub total_frames: u64,
    pub steps: Vec<StepTimingInfo>,
}

#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct StepTimingInfo {
    pub name: String,
    pub frames: u64,
    pub ms_per_frame: f64,
    pub fps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterParams {
    pub denoise_radius: Option<u32>,
    pub denoise_model_id: Option<String>,
    pub deblur_amount: Option<f32>,
    pub deblur_model_id: Option<String>,
    pub dedup_threshold: Option<f32>,
    pub ffmpeg_filter: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    pub scale: Option<u32>,
    pub model_id: Option<String>,
    pub resize: Option<f32>,
    pub filter: Option<FilterParams>,
    pub decompress_model_id: Option<String>,
    pub output_resize: Option<f32>,
    pub fps_multiplier: Option<u32>,
    pub interp_model: Option<String>,
    pub ffmpeg_args: Option<Vec<String>>,
    pub tonemap: Option<String>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

pub fn filter_to_core(f: FilterParams) -> core::FilterConfig {
    core::FilterConfig {
        denoise_radius: f.denoise_radius,
        denoise_model_id: f.denoise_model_id,
        deblur_amount: f.deblur_amount,
        deblur_model_id: f.deblur_model_id,
        dedup_threshold: f.dedup_threshold,
        ffmpeg_filter: f.ffmpeg_filter,
    }
}
