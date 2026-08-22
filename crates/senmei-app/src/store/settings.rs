use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::data_dir;

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub language: String,
    pub theme: String,
    /// Action-id → key-combo overrides (e.g. `render: "Ctrl+Shift+R"`);
    /// absent entries use the app defaults.
    #[serde(default)]
    pub hotkeys: Option<HashMap<String, String>>,
    /// Fused RGB8 tile size in px; `None` = engine default (640).
    #[serde(default)]
    pub tile_size: Option<u32>,
    /// Readback pipeline depth (batches kept in flight); `None` = 1. More
    /// depth overlaps the readback with more GPU forwards (higher utilisation,
    /// more VRAM + cancel latency).
    #[serde(default)]
    pub pipeline_depth: Option<u32>,
    /// Preferred inference backend; `None` = auto (libtorch if compiled, else Vulkan).
    #[serde(default)]
    pub backend: Option<senmei_ml::EngineBackend>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: "en".into(),
            theme: "dark".into(),
            hotkeys: None,
            tile_size: None,
            pipeline_depth: None,
            backend: None,
        }
    }
}

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

pub fn load_settings() -> Settings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_settings(settings: &Settings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}
