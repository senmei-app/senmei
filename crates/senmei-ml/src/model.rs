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
    pub download_url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    /// Whether an engine can load these weights yet (e.g. a `.pth` state dict
    /// that still needs TorchScript conversion is not loadable).
    #[serde(default = "default_true")]
    pub loadable: bool,
    #[serde(default)]
    #[specta(skip)]
    pub metadata: serde_json::Value,
}

fn default_true() -> bool {
    true
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
                let value: serde_json::Value = serde_json::from_str(&json)?;
                if value.is_array() {
                    let models: Vec<ModelMetadata> = serde_json::from_value(value)?;
                    self.models.extend(models);
                } else {
                    self.models.push(serde_json::from_value(value)?);
                }
            }
        }
        Ok(())
    }

    pub fn models(&self) -> &[ModelMetadata] {
        &self.models
    }

    /// Resolve a model by id to a `ModelRef` pointing at its torch weight file.
    pub fn resolve(&self, id: &str, dir: &Path) -> Option<ModelRef> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .and_then(|m| m.torch.as_ref())
            .map(|f| ModelRef {
                id: id.to_string(),
                path: dir.join(f),
            })
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
        // `loadable` defaults to true when absent.
        assert!(registry.models()[0].loadable);
    }

    #[test]
    fn registry_parses_loadable_false() {
        let json = r#"[
            {"id": "span", "kind": "upscale", "scale": 4, "arch": "span", "loadable": false}
        ]"#;
        let registry = Registry::from_json(json).unwrap();
        assert!(!registry.models()[0].loadable);
    }

    #[test]
    fn registry_loads_example_metadata() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = Registry::new();
        registry.load_dir(path).unwrap();
        assert_eq!(registry.models().len(), 4);
        assert_eq!(registry.models()[0].id, "rife-4.26");
        assert!(matches!(registry.models()[0].kind, ModelKind::Interpolate));
        assert_eq!(registry.models()[0].torch.as_deref(), Some("rife-4.26.pt"));
        assert_eq!(registry.models()[1].id, "realesrgan-x4plus");
        assert!(matches!(registry.models()[1].kind, ModelKind::Upscale));
        assert_eq!(registry.models()[1].scale, 4);
        assert_eq!(registry.models()[2].id, "realesrgan-x4plus-anime");
        assert_eq!(registry.models()[3].id, "realesrgan-x2plus");
        assert_eq!(registry.models()[3].scale, 2);
    }

    #[test]
    fn registry_resolves_model_ref() {
        let json = r#"[
            {"id": "span", "kind": "upscale", "scale": 4, "arch": "span", "torch": "span.pt"}
        ]"#;
        let registry = Registry::from_json(json).unwrap();
        let mref = registry.resolve("span", Path::new("/models")).unwrap();
        assert_eq!(mref.id, "span");
        assert_eq!(mref.path, Path::new("/models/span.pt"));
        assert!(registry.resolve("missing", Path::new("/models")).is_none());
    }
}
