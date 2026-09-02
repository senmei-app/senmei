//! Model weight download + f16 burnpack conversion (`render` feature).

use super::load_registry;

/// Download a model's weights (`.pth`/`.onnx`/ncnn `.bin`, sha256-verified when
/// pinned) and convert to the f16 `.bpk` burnpack. Handles RIFE's ncnn release
/// zips (extract one entry) and skips when the target already exists. Mirrors
/// the GUI's `download_model` without Tauri. Needs the `render` feature (burn
/// convert). `on_progress` receives (downloaded, total) bytes.
#[cfg(feature = "render")]
pub fn download_model(
    model_id: &str,
    mut on_progress: impl FnMut(u64, u64) + Send,
) -> Result<String, String> {
    let (registry, dir) = load_registry()?;
    let meta = registry
        .models()
        .iter()
        .find(|m| m.id == model_id)
        .cloned()
        .ok_or_else(|| format!("model not found: {model_id}"))?;
    let resolved = registry.resolve(model_id, &dir);
    let convert_arg = resolved
        .as_ref()
        .map(|m| match m.arch.as_str() {
            "span" => m.feature_channels,
            "srvgg" => m.num_conv,
            _ => m.num_block,
        })
        .unwrap_or(4);
    let layer_norm = resolved.as_ref().map(|m| m.layer_norm).unwrap_or(false);
    let dysample = resolved.as_ref().map(|m| m.dysample).unwrap_or(true);
    let shuffle = resolved.as_ref().map(|m| m.shuffle).unwrap_or(1);
    if meta.license_blocked() {
        return Err(format!(
            "model {model_id} has an unconfirmed/restrictive license ({}); refusing download",
            meta.license.as_deref().unwrap_or("none")
        ));
    }
    if !meta.loadable {
        return Err(format!("model {model_id} has no loadable arch yet"));
    }
    let url = meta
        .download_url
        .clone()
        .ok_or_else(|| format!("model {model_id} has no download_url"))?;
    let weight = meta
        .weights
        .as_ref()
        .and_then(|w| w.first())
        .cloned()
        .ok_or_else(|| format!("model {model_id} has no weights"))?;
    let is_ncnn = weight.ends_with(".bin");
    if !weight.ends_with(".bpk") && !is_ncnn {
        return Err(format!(
            "expected f16 burnpack or ncnn weight, got {weight}"
        ));
    }
    // Weights are plain filenames in the models dir — never path components.
    if std::path::Path::new(&weight).components().count() != 1 {
        return Err(format!("unsafe weight path in metadata: {weight}"));
    }
    let is_archive = url.ends_with(".zip");
    // Multi-model archives (e.g. the nihui rife release zip bundles every
    // version) need a version-specific entry; default to the weight filename.
    let extract_suffix = meta
        .metadata
        .get("extract_suffix")
        .and_then(|v| v.as_str())
        .unwrap_or(&weight)
        .to_string();
    let target = dir.join(&weight);
    if target.is_file() {
        log::info!("download_model: {model_id} already present, skipping");
        return Ok(target.to_string_lossy().into_owned());
    }
    let onnx = std::path::Path::new(&url)
        .extension()
        .and_then(|e| e.to_str())
        == Some("onnx");
    let st = std::path::Path::new(&url)
        .extension()
        .and_then(|e| e.to_str())
        == Some("safetensors");
    let ext = if onnx {
        "onnx"
    } else if st {
        "safetensors"
    } else if is_archive {
        "zip"
    } else {
        "pth"
    };
    log::info!("download_model: {model_id} <- {url} -> {}", dir.display());
    let base = weight.trim_end_matches(".f16.bpk");
    let source = senmei_media::download_to_temp(
        &url,
        &dir,
        &format!("{base}.{ext}"),
        meta.sha256.as_deref(),
        &mut on_progress,
    )
    .map_err(|e| {
        log::error!("download_model {model_id}: download failed: {e}");
        e.to_string()
    })?;
    log::info!(
        "download_model: {model_id} downloaded to {}",
        source.display()
    );
    // RIFE ships ncnn weights: the .bin is inside a release zip, or a raw
    // .bin — either way no burnpack conversion.
    if is_archive {
        senmei_media::extract_binary(&source, &target, &extract_suffix).map_err(|e| {
            log::error!("download_model {model_id}: extract failed: {e}");
            e.to_string()
        })?;
        let _ = std::fs::remove_file(&source);
        log::info!("download_model: {model_id} wrote {}", target.display());
        return Ok(target.to_string_lossy().into_owned());
    }
    if is_ncnn {
        std::fs::rename(&source, &target).map_err(|e| {
            log::error!("download_model {model_id}: rename failed: {e}");
            e.to_string()
        })?;
        log::info!("download_model: {model_id} wrote {}", target.display());
        return Ok(target.to_string_lossy().into_owned());
    }
    let conv = if onnx {
        senmei_ml::convert_onnx_to_bpk(
            &meta.arch,
            &source,
            &target,
            meta.scale,
            convert_arg,
            shuffle,
        )
    } else if st {
        senmei_ml::convert_safetensors_to_bpk(&meta.arch, &source, &target, meta.scale, convert_arg)
    } else {
        senmei_ml::convert_pth_to_bpk(&senmei_ml::ConvertOptions {
            arch: meta.arch.as_str(),
            pth_path: source.as_path(),
            bpk_path: target.as_path(),
            scale: meta.scale,
            num_block: convert_arg,
            layer_norm,
            dysample,
            shuffle,
        })
    };
    if let Err(e) = conv {
        log::error!("download_model {model_id}: conversion failed: {e}");
        return Err(e.to_string());
    }
    let _ = std::fs::remove_file(&source);
    log::info!("download_model: {model_id} wrote {}", target.display());
    Ok(target.to_string_lossy().into_owned())
}
