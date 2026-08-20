//! Model registry helpers for the commands layer.

use std::path::{Path, PathBuf};

/// Writable models dir for a packaged app (catalog materialized by
/// [`ensure_catalog`]).
pub fn data_models_dir() -> PathBuf {
    crate::store::data_dir().join("models")
}

/// Resolve the models dir. Dev uses the repo checkout `models/` (keeps
/// pre-converted `.bpk` and gitignored weights local); a packaged app uses the
/// writable data dir once [`ensure_catalog`] has materialized `metadata.json`.
pub fn models_dir() -> PathBuf {
    // Dev anchor: cargo tauri dev runs the binary from the crate dir, so
    // CWD-relative paths can miss models/ at the repo root.
    let anchored = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    if anchored.join("metadata.json").is_file() {
        return anchored;
    }
    let data_models = data_models_dir();
    if data_models.join("metadata.json").is_file() {
        return data_models;
    }
    for dir in [PathBuf::from("models"), PathBuf::from("../models")] {
        if dir.join("metadata.json").is_file() {
            return dir;
        }
    }
    data_models
}

/// Find `metadata.json` under a resource dir. Tauri bundles resource paths
/// with `..` components as `_up_` dirs, so the catalog may be nested.
fn find_metadata_json(root: &Path) -> Option<PathBuf> {
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 8 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push((p, depth + 1));
            } else if p.file_name().map(|n| n == "metadata.json").unwrap_or(false) {
                return Some(p);
            }
        }
    }
    None
}

/// Materialize the model catalog (`metadata.json`) into the writable data dir
/// so a packaged app finds it without a repo checkout. Source: bundled resource
/// dir (release) or the dev repo checkout. Idempotent.
pub fn ensure_catalog(resource_dir: Option<&Path>) -> Result<PathBuf, String> {
    let dir = data_models_dir();
    let target = dir.join("metadata.json");
    if let Some(source) = resource_dir.and_then(find_metadata_json).or_else(|| {
        let anchored =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/metadata.json");
        anchored.is_file().then_some(anchored)
    }) {
        // Refresh when the bundled catalog differs (e.g. after an app update);
        // a stale data-dir copy would otherwise hide newly added models.
        if std::fs::read(&target).ok() != std::fs::read(&source).ok() {
            std::fs::create_dir_all(&dir)
                .and_then(|_| std::fs::copy(&source, &target))
                .map_err(|e| format!("failed to materialize model catalog: {e}"))?;
        }
    } else if !target.is_file() {
        return Err(
            "model catalog (metadata.json) not found in resources or repo checkout".to_string(),
        );
    }
    Ok(dir)
}

pub fn load_registry() -> Result<(senmei_ml::Registry, PathBuf), String> {
    let dir = models_dir();
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&dir).map_err(|e| e.to_string())?;
    Ok((registry, dir))
}

pub fn engine_for_model(model_id: &str) -> Result<Box<dyn senmei_ml::InferenceEngine>, String> {
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("model not found: {model_id}"))?;
    if meta.license_blocked() {
        return Err(format!(
            "model {model_id} has an unconfirmed/restrictive license ({}); refusing to load weights",
            meta.license.as_deref().unwrap_or("none")
        ));
    }
    if !meta.loadable {
        return Err(format!("model {model_id} has no loadable weights yet"));
    }
    let mref = registry
        .resolve(model_id, &dir)
        .ok_or_else(|| format!("model weights not resolved: {model_id}"))?;
    let backend = crate::store::load_settings()
        .backend
        .unwrap_or(senmei_ml::EngineBackend::Auto);
    let mut engine = senmei_ml::engine_for_model(&mref, backend, &crate::store::data_dir())
        .map_err(|e| e.to_string())?;
    engine.load(&mref).map_err(|e| e.to_string())?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_data_dir(name: &str, test: impl FnOnce()) {
        let _guard = crate::store::TEST_ENV_LOCK.lock().unwrap();
        let base = std::env::temp_dir()
            .join(format!("senmei-models-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::env::set_var("XDG_DATA_HOME", &base);
        test();
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn ensure_catalog_materializes_into_data_dir() {
        with_temp_data_dir("catalog", || {
            ensure_catalog(None).unwrap();
            assert!(data_models_dir().join("metadata.json").is_file());
        });
    }

    #[test]
    fn ensure_catalog_is_idempotent() {
        with_temp_data_dir("idempotent", || {
            ensure_catalog(None).unwrap();
            let dir = ensure_catalog(None).unwrap();
            assert!(dir.join("metadata.json").is_file());
        });
    }

    #[test]
    fn find_catalog_handles_tauri_up_mangling() {
        // Tauri bundles resource paths with `..` as `_up_` dirs; the recursive
        // find must still locate the catalog there.
        let res = std::env::temp_dir().join(format!("senmei-res-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&res);
        let nested = res.join("_up_").join("_up_").join("models");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("metadata.json"), b"{}").unwrap();

        let found = find_metadata_json(&res);
        assert!(found.is_some(), "nested catalog not found");
        assert_eq!(found.unwrap().file_name().unwrap(), "metadata.json");
        let _ = std::fs::remove_dir_all(&res);
    }
}
