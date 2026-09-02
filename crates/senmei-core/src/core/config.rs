//! Render-config shapes + enriched settings schema (compiled without `render`).

use super::list_models;

// `FilterConfig`/`RenderConfig` are plain (de)serializable config shapes used by
// both transports; keep them compiled regardless of the `render` feature so
// `senmei-server` builds without it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct FilterConfig {
    /// Denoise box-blur radius; `denoise_model_id` overrides with a learned model.
    pub denoise_radius: Option<u32>,
    /// Learned denoise model id (kind=denoise); empty = reference filter only.
    pub denoise_model_id: Option<String>,
    /// Deblur unsharp-mask amount; `deblur_model_id` overrides with a learned model.
    pub deblur_amount: Option<f32>,
    /// Learned deblur model id (kind=deblur); empty = reference filter only.
    pub deblur_model_id: Option<String>,
    /// Dedup mean-pixel-diff threshold in [0,1]; drops near-duplicate frames.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub dedup_threshold: Option<f32>,
    /// Free-form FFmpeg `-vf` filter graph applied per frame (e.g.
    /// `"hue=h=10,unsharp"`). Frame-preserving only (1:1) — filters that change
    /// the output size are rejected. Runs between the reference/ML filters and
    /// the final `output_resize`.
    pub ffmpeg_filter: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", default)]
pub struct RenderConfig {
    /// Input video path (required).
    pub input: String,
    /// Output video path (required); container guessed from extension.
    pub output: String,
    /// Integer upscale factor for the upscale step (1..=4).
    #[schemars(range(min = 1, max = 4))]
    pub scale: Option<u32>,
    /// Upscale model id (kind=upscale, license-gated); empty = reference upscale only.
    pub model_id: Option<String>,
    /// Decompress model id (scale-1 de-artifact pass, e.g. RealPLKSR 1×).
    pub decompress_model_id: Option<String>,
    /// Pre-upscale resize factor (>0, e.g. 0.5 to shrink first).
    pub resize: Option<f32>,
    /// Optional filter chain: denoise / deblur / dedup.
    pub filter: Option<FilterConfig>,
    /// Post-upscale resize factor (>0, e.g. 0.5 for a net 1x).
    pub output_resize: Option<f32>,
    /// Frame-rate multiplier for interpolation (1..=16).
    #[schemars(range(min = 1, max = 16))]
    pub fps_multiplier: Option<u32>,
    /// Interpolation model id (kind=interpolate); empty = linear blend.
    pub interp_model: Option<String>,
    /// Extra raw ffmpeg output args; appended last, override structured fields.
    pub ffmpeg_args: Option<Vec<String>>,
    /// HDR→SDR tonemapping for decode: "auto" | "always" | "off".
    pub tonemap: Option<String>,
    /// Render only from this timestamp (ms); pairs with `end_ms` for samples.
    pub start_ms: Option<u64>,
    /// Render only up to this timestamp (ms).
    pub end_ms: Option<u64>,
}

/// Enriched settings schema for agents — works without the `render` feature.
/// Returns the render-config JSON Schema (schemars), the model slots (which
/// registry models fill which config field) and the hard constraints (license
/// gate, ranges, confirm gate).
pub fn settings_schema() -> serde_json::Value {
    let config_schema = schemars::schema_for!(RenderConfig);
    let models = list_models();

    let slot = |field: &str, kind: &str| -> serde_json::Value {
        let kind_v = serde_json::Value::String(kind.to_string());
        let candidates: Vec<serde_json::Value> = models
            .iter()
            .filter(|m| serde_json::to_value(m.kind).ok() == Some(kind_v.clone()))
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "kind": m.kind,
                    "scale": m.scale,
                    "arch": m.arch,
                    "loadable": m.loadable,
                    "license": m.license,
                    "licenseBlocked": m.license_blocked(),
                })
            })
            .collect();
        serde_json::json!({ "field": field, "kind": kind, "models": candidates })
    };

    serde_json::json!({
        "renderConfig": serde_json::to_value(&config_schema).unwrap_or_default(),
        "modelSlots": [
            slot("model_id", "upscale"),
            slot("interp_model", "interpolate"),
            slot("filter.denoise_model_id", "denoise"),
            slot("filter.deblur_model_id", "deblur"),
        ],
        "constraints": {
            "scale": "1..=4",
            "fpsMultiplier": "1..=16",
            "tonemap": ["auto", "always", "off"],
            "licenseGate": "every referenced model must be permissive + loadable (licenseBlocked/!loadable rejected)",
            "confirmGate": "full render starts only after propose_render + confirm_render",
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_schema_has_slots_and_constraints() {
        let schema = settings_schema();
        let obj = schema.as_object().expect("schema is an object");
        assert!(obj.contains_key("renderConfig"));
        assert!(obj.contains_key("constraints"));

        let slots = obj["modelSlots"]
            .as_array()
            .expect("modelSlots is an array");
        assert_eq!(slots.len(), 4);
        for slot in slots {
            assert!(slot.get("field").is_some());
            assert!(slot.get("kind").is_some());
            assert!(slot.get("models").is_some());
        }

        let upscale = slots
            .iter()
            .find(|s| s["field"] == "model_id")
            .expect("upscale slot present");
        assert_eq!(upscale["kind"], "upscale");
        let ids: Vec<&str> = upscale["models"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(
            ids.contains(&"span-2x-nomosuni-ldl"),
            "SPAN registered: {ids:?}"
        );
    }
}
