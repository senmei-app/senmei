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
    /// Weight files (e.g. `.pth`, `.bpk`), first entry is the primary.
    #[serde(default)]
    pub weights: Option<Vec<String>>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub source_url: Option<String>,
    /// Direct download URL for the primary weight file (download-on-demand).
    #[serde(default)]
    pub download_url: Option<String>,
    /// SHA-256 of the primary weight file, verified on download.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Whether the engine can load these weights yet.
    #[serde(default = "default_true")]
    pub loadable: bool,
    #[serde(default)]
    #[specta(skip)]
    pub metadata: serde_json::Value,
}

fn default_true() -> bool {
    true
}

impl ModelMetadata {
    /// Whether the license blocks download/use. Blocks pending-review markers
    /// (`verify`/`unclear`) and copyleft / non-commercial licenses regardless
    /// of the `loadable` flag — the review gate must never auto-unlock an
    /// unclear license. Missing license → blocked.
    pub fn license_blocked(&self) -> bool {
        let Some(lic) = self.license.as_deref() else {
            return true;
        };
        let lic = lic.to_ascii_lowercase();
        [
            "verify", "unclear", "gpl", "lgpl", "agpl", "cc-by-nc", "cc-by-nd", "cc-by-sa",
        ]
        .iter()
        .any(|b| lic.contains(b))
    }
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub id: String,
    pub arch: String,
    pub scale: u32,
    /// RRDB blocks for the `realesrgan` arch family.
    pub num_block: u32,
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

    /// Resolve a model by id to a `ModelRef` pointing at its primary weight
    /// file (e.g. the `.bpk`); carries the arch/config the engine needs.
    pub fn resolve(&self, id: &str, dir: &Path) -> Option<ModelRef> {
        self.models.iter().find(|m| m.id == id).and_then(|m| {
            m.weights
                .as_ref()
                .and_then(|f| f.first())
                .map(|f| ModelRef {
                    id: id.to_string(),
                    arch: m.arch.clone(),
                    scale: m.scale,
                    num_block: m
                        .metadata
                        .get("num_block")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4) as u32,
                    path: dir.join(f),
                })
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
        let registry = Registry {
            models: serde_json::from_str(json).unwrap(),
        };
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
        let registry = Registry {
            models: serde_json::from_str(json).unwrap(),
        };
        assert!(!registry.models()[0].loadable);
    }

    #[test]
    fn registry_loads_example_metadata() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let mut registry = Registry::new();
        registry.load_dir(path).unwrap();
        assert_eq!(registry.models().len(), 17);
        assert_eq!(registry.models()[0].id, "fallin-soft");
        assert!(registry.models()[0].loadable);
        assert_eq!(registry.models()[1].id, "fallin-strong");
        assert!(registry.models()[1].loadable);
        assert_eq!(registry.models()[2].id, "4x-alchemy");
        assert!(registry.models()[2].loadable);
        assert_eq!(registry.models()[3].id, "real-cugan-x2");
        assert!(matches!(registry.models()[3].kind, ModelKind::Upscale));
        assert_eq!(registry.models()[3].scale, 2);
        assert!(registry.models()[3].loadable);
        assert_eq!(registry.models()[4].id, "realesrgan-animevideo-x2");
        assert_eq!(registry.models()[5].id, "realesrgan-animevideo-x4");
        assert_eq!(registry.models()[6].id, "realesrgan-x4plus-anime");
        assert_eq!(registry.models()[6].scale, 4);
        assert_eq!(registry.models()[7].id, "bsrgan");
        assert!(matches!(registry.models()[7].kind, ModelKind::Upscale));
        assert_eq!(registry.models()[7].scale, 4);
        assert_eq!(registry.models()[7].arch, "realesrgan");
        assert!(registry.models()[7].loadable);
        assert_eq!(registry.models()[7].license.as_deref(), Some("MIT"));
        assert!(matches!(registry.models()[8].kind, ModelKind::Denoise));
        assert_eq!(registry.models()[9].id, "real-plksr-deh264");
        assert_eq!(registry.models()[9].sha256.as_deref().unwrap().len(), 64);
        assert_eq!(registry.models()[10].id, "real-plksr-dejpg");
        assert_eq!(registry.models()[11].id, "anime1080fixer");
        assert_eq!(registry.models()[12].id, "drunet-color");
        assert!(matches!(registry.models()[12].kind, ModelKind::Denoise));
        assert_eq!(registry.models()[12].arch, "drunet");
        assert!(registry.models()[12].loadable);
        assert_eq!(registry.models()[12].license.as_deref(), Some("MIT"));
        assert_eq!(registry.models()[13].id, "nafnet-gopro-width32");
        assert!(matches!(registry.models()[13].kind, ModelKind::Deblur));
        assert_eq!(registry.models()[13].arch, "nafnet");
        assert!(registry.models()[13].loadable);
        assert_eq!(registry.models()[13].license.as_deref(), Some("MIT"));
        assert_eq!(registry.models()[14].id, "rife-v4.6");
        assert!(matches!(registry.models()[14].kind, ModelKind::Interpolate));
        assert_eq!(registry.models()[14].arch, "rife46");
        assert!(registry.models()[14].loadable);
        assert_eq!(registry.models()[15].id, "ifrnet-vimeo90k");
        assert_eq!(registry.models()[16].id, "ifrnet-gopro");
        assert!(matches!(registry.models()[15].kind, ModelKind::Interpolate));
        assert_eq!(registry.models()[15].arch, "ifrnet");
        assert!(registry.models()[15].loadable);
        assert_eq!(registry.models()[15].license.as_deref(), Some("MIT"));
    }

    #[test]
    fn registry_resolves_model_ref() {
        let json = r#"[
            {"id": "span", "kind": "upscale", "scale": 4, "arch": "realesrgan",
             "weights": ["span.pth"], "metadata": {"num_block": 6}}
        ]"#;
        let registry = Registry {
            models: serde_json::from_str(json).unwrap(),
        };
        let mref = registry.resolve("span", Path::new("/models")).unwrap();
        assert_eq!(mref.id, "span");
        assert_eq!(mref.arch, "realesrgan");
        assert_eq!(mref.scale, 4);
        assert_eq!(mref.num_block, 6);
        assert_eq!(mref.path, Path::new("/models/span.pth"));
        assert!(registry.resolve("missing", Path::new("/models")).is_none());
    }

    #[test]
    fn license_gate_blocks_unclear_and_copyleft() {
        let mk = |license: Option<&str>| ModelMetadata {
            id: "m".into(),
            kind: ModelKind::Upscale,
            scale: 2,
            arch: "upcunet2x".into(),
            weights: None,
            license: license.map(str::to_string),
            source_url: None,
            download_url: None,
            sha256: None,
            loadable: true, // loadable alone must NOT unlock (review gate)
            metadata: serde_json::Value::Null,
        };
        assert!(mk(Some("verify (cszn)")).license_blocked());
        assert!(mk(Some("unclear")).license_blocked());
        assert!(mk(Some("GPL-3.0")).license_blocked());
        assert!(mk(Some("CC-BY-NC-4.0")).license_blocked());
        assert!(mk(None).license_blocked());
        assert!(!mk(Some("MIT")).license_blocked());
        assert!(!mk(Some("Apache-2.0")).license_blocked());
        assert!(!mk(Some("CC-BY-4.0")).license_blocked());
    }
}
