use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Vulkan,
}

#[derive(Debug, Clone, Copy)]
pub struct EngineCaps {
    pub backend: Backend,
    pub half: bool,
    pub tiles: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct InferOptions {
    pub half: bool,
    pub tile_size: Option<u32>,
}

pub trait InferenceEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> EngineCaps;
    fn load(&mut self, model: &ModelRef) -> Result<()>;
    fn infer(&mut self, input: &Tensor, opts: &InferOptions) -> Result<Tensor>;
}

/// Run an engine over a full input, tiling when the engine advertises tile support
/// and the input exceeds `opts.tile_size`. Tile outputs are stitched with overlap
/// averaging; the output canvas is scaled by the engine's per-tile scale factor.
pub fn infer_tiled(
    engine: &mut dyn InferenceEngine,
    input: &Tensor,
    opts: &InferOptions,
) -> Result<Tensor> {
    let caps = engine.capabilities();
    let Some(tile_size) = opts.tile_size else {
        return engine.infer(input, opts);
    };
    if !caps.tiles {
        return engine.infer(input, opts);
    }
    let tile_size = tile_size as usize;
    let h = input.shape[2];
    let w = input.shape[3];
    if h <= tile_size && w <= tile_size {
        return engine.infer(input, opts);
    }

    let overlap = tile_size / 4;
    let tiles = crate::tile(input, tile_size, overlap);
    let mut out_tiles = Vec::with_capacity(tiles.len());
    for (x, y, t) in &tiles {
        let out = engine.infer(t, opts)?;
        out_tiles.push((*x, *y, out));
    }

    let scale_h = out_tiles[0].2.shape[2] as f32 / tiles[0].2.shape[2] as f32;
    let scale_w = out_tiles[0].2.shape[3] as f32 / tiles[0].2.shape[3] as f32;
    let out_h = (h as f32 * scale_h).round() as usize;
    let out_w = (w as f32 * scale_w).round() as usize;
    let scaled: Vec<(usize, usize, Tensor)> = out_tiles
        .iter()
        .map(|(x, y, t)| {
            let sx = (*x as f32 * scale_w).round() as usize;
            let sy = (*y as f32 * scale_h).round() as usize;
            (sx, sy, t.clone())
        })
        .collect();
    Ok(crate::stitch(&scaled, out_h, out_w, input.shape[1]))
}

/// Pick an engine for a model based on its weight-file format.
pub fn engine_for_model(model: &ModelRef) -> Result<Box<dyn InferenceEngine>> {
    let ext = model.path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "param" | "bin" => Ok(Box::new(NcnnEngine::new())),
        _ => Err(Error::new(format!("unsupported model format: {ext}"))),
    }
}

pub struct NcnnEngine {
    engine: Option<senmei_ncnn::Engine>,
    gpu: bool,
}

impl NcnnEngine {
    pub fn new() -> Self {
        Self::with_gpu(true)
    }

    /// `gpu` requests the Vulkan backend; the shim falls back to CPU when no
    /// Vulkan device is available.
    pub fn with_gpu(gpu: bool) -> Self {
        Self { engine: None, gpu }
    }
}

impl InferenceEngine for NcnnEngine {
    fn name(&self) -> &'static str {
        "ncnn"
    }

    fn capabilities(&self) -> EngineCaps {
        EngineCaps {
            backend: if self.gpu { Backend::Vulkan } else { Backend::Cpu },
            half: true,
            tiles: true,
        }
    }

    fn load(&mut self, model: &ModelRef) -> Result<()> {
        let bin = model.path.with_extension("bin");
        let engine =
            senmei_ncnn::Engine::new(self.gpu).map_err(|e| Error::new(e.to_string()))?;
        engine
            .load(&model.path, &bin)
            .map_err(|e| Error::new(e.to_string()))?;
        self.engine = Some(engine);
        Ok(())
    }

    fn infer(&mut self, input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| Error::new("model not loaded"))?;
        let h = input.shape[2];
        let w = input.shape[3];
        let (data, oh, ow) = engine
            .infer(&input.data, h, w)
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Tensor::new(vec![1, 3, oh, ow], data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(h: usize, w: usize) -> Tensor {
        let mut data = vec![0f32; 3 * h * w];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i % 251) as f32 / 251.0;
        }
        Tensor::new(vec![1, 3, h, w], data)
    }

    struct IdentityEngine;

    impl InferenceEngine for IdentityEngine {
        fn name(&self) -> &'static str {
            "identity-test"
        }
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { backend: Backend::Cpu, half: false, tiles: true }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            Ok(input.clone())
        }
    }

    struct ScaleEngine {
        factor: usize,
    }

    impl InferenceEngine for ScaleEngine {
        fn name(&self) -> &'static str {
            "scale-test"
        }
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { backend: Backend::Cpu, half: false, tiles: true }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            let h = input.shape[2];
            let w = input.shape[3];
            Ok(crate::bilinear(input, h * self.factor, w * self.factor))
        }
    }

    struct NoTilesEngine;

    impl InferenceEngine for NoTilesEngine {
        fn name(&self) -> &'static str {
            "notiles-test"
        }
        fn capabilities(&self) -> EngineCaps {
            EngineCaps { backend: Backend::Cpu, half: false, tiles: false }
        }
        fn load(&mut self, _m: &ModelRef) -> Result<()> {
            Ok(())
        }
        fn infer(&mut self, input: &Tensor, _o: &InferOptions) -> Result<Tensor> {
            Ok(input.clone())
        }
    }

    #[test]
    fn tiled_identity_reconstructs() {
        let t = input(16, 16);
        let mut engine = IdentityEngine;
        let opts = InferOptions { half: false, tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, t.shape);
        assert_eq!(out.data, t.data);
    }

    #[test]
    fn tiled_scaled_dims() {
        let t = input(16, 16);
        let mut engine = ScaleEngine { factor: 2 };
        let opts = InferOptions { half: false, tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, vec![1, 3, 32, 32]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn small_input_skips_tiling() {
        let t = input(8, 8);
        let mut engine = ScaleEngine { factor: 2 };
        let opts = InferOptions { half: false, tile_size: Some(16) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.shape, vec![1, 3, 16, 16]);
    }

    #[test]
    fn engine_without_tiling_goes_whole() {
        let t = input(16, 16);
        let mut engine = NoTilesEngine;
        let opts = InferOptions { half: false, tile_size: Some(8) };
        let out = infer_tiled(&mut engine, &t, &opts).unwrap();
        assert_eq!(out.data, t.data);
    }

    #[test]
    fn factory_picks_engine_by_format() {
        let ncnn = crate::model::ModelRef {
            id: "rife".into(),
            path: std::path::PathBuf::from("/models/rife.param"),
        };
        assert_eq!(engine_for_model(&ncnn).unwrap().name(), "ncnn");

        let bad = crate::model::ModelRef {
            id: "x".into(),
            path: std::path::PathBuf::from("/models/x.onnx"),
        };
        assert!(engine_for_model(&bad).is_err());
    }

    #[test]
    #[ignore = "requires models/up2x-no-denoise.{param,bin} (from realcugan-ncnn-vulkan)"]
    fn ncnn_loads_real_model() {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let param = dir.join("up2x-no-denoise.param");
        if !param.exists() {
            eprintln!("model files missing, skipping");
            return;
        }
        let mut engine = NcnnEngine::with_gpu(false);
        engine
            .load(&ModelRef { id: "real-cugan-x2".into(), path: param })
            .unwrap();
        let input = Tensor::new(vec![1, 3, 64, 64], vec![0.5f32; 3 * 64 * 64]);
        let out = engine
            .infer(&input, &InferOptions { half: false, tile_size: None })
            .unwrap();
        // Real-CUGAN upcunet crops a fixed border: out = 2*h - 72.
        assert_eq!(out.shape, vec![1, 3, 56, 56]);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }
}
