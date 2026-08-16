use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Interpolate,
    Upscale,
    Denoise,
    Decompress,
    Deblur,
}

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct ModelMetadata {
    pub id: String,
    pub kind: ModelKind,
    #[serde(default)]
    pub scale: u32,
    pub arch: String,
    #[serde(default)]
    pub torch: Option<String>,
    #[serde(default)]
    pub ncnn: Option<Vec<String>>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    #[specta(skip)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub id: String,
    pub path: PathBuf,
}

#[derive(Debug, Default)]
pub struct Registry {
    models: Vec<ModelMetadata>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(Self {
            models: serde_json::from_str(json)?,
        })
    }

    pub fn load_dir(&mut self, dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let json = std::fs::read_to_string(&path)?;
                self.models.push(serde_json::from_str(&json)?);
            }
        }
        Ok(())
    }

    pub fn models(&self) -> &[ModelMetadata] {
        &self.models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_from_json_array() {
        let json = r#"[
            {"id": "rife-v4", "kind": "interpolate", "scale": 1, "arch": "rife425"},
            {"id": "span", "kind": "upscale", "scale": 4, "arch": "span"}
        ]"#;
        let registry = Registry::from_json(json).unwrap();
        assert_eq!(registry.models().len(), 2);
        assert_eq!(registry.models()[0].id, "rife-v4");
        assert!(matches!(registry.models()[0].kind, ModelKind::Interpolate));
        assert_eq!(registry.models()[1].scale, 4);
    }

    #[test]
    fn registry_loads_example_metadata() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = Registry::new();
        registry.load_dir(path).unwrap();
        assert_eq!(registry.models().len(), 1);
        assert_eq!(registry.models()[0].id, "rife-4.26");
        assert!(matches!(registry.models()[0].kind, ModelKind::Interpolate));
        assert_eq!(registry.models()[0].torch.as_deref(), Some("rife-4.26.pt"));
    }
}
