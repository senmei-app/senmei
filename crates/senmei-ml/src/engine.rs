use crate::model::ModelRef;
use crate::tensor::Tensor;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
    Mps,
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

#[cfg(not(feature = "torch"))]
pub struct TorchEngine;

#[cfg(not(feature = "torch"))]
impl TorchEngine {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(feature = "torch"))]
impl InferenceEngine for TorchEngine {
    fn name(&self) -> &'static str {
        "torch"
    }

    fn capabilities(&self) -> EngineCaps {
        EngineCaps {
            backend: Backend::Cuda,
            half: true,
            tiles: true,
        }
    }

    fn load(&mut self, _model: &ModelRef) -> Result<()> {
        Err(Error::unimplemented("TorchEngine::load (enable the `torch` feature)"))
    }

    fn infer(&mut self, _input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        Err(Error::unimplemented("TorchEngine::infer (enable the `torch` feature)"))
    }
}

#[cfg(feature = "torch")]
pub struct TorchEngine {
    model: Option<tch::CModule>,
    device: tch::Device,
}

#[cfg(feature = "torch")]
impl TorchEngine {
    pub fn new() -> Self {
        Self {
            model: None,
            device: tch::Device::Cpu,
        }
    }
}

#[cfg(feature = "torch")]
impl InferenceEngine for TorchEngine {
    fn name(&self) -> &'static str {
        "torch"
    }

    fn capabilities(&self) -> EngineCaps {
        EngineCaps {
            backend: Backend::Cpu,
            half: false,
            tiles: true,
        }
    }

    fn load(&mut self, model: &ModelRef) -> Result<()> {
        let module = tch::CModule::load(&model.path).map_err(|e| Error::new(e.to_string()))?;
        self.model = Some(module);
        Ok(())
    }

    fn infer(&mut self, input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        let model = self.model.as_ref().ok_or_else(|| Error::new("model not loaded"))?;
        let h = input.shape[2] as i64;
        let w = input.shape[3] as i64;
        let t = tch::Tensor::from_slice(&input.data)
            .view([1, 3, h, w])
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .contiguous();
        let out = model.forward_ts(&[t]).map_err(|e| Error::new(e.to_string()))?;
        let oh = out.size()[2] as usize;
        let ow = out.size()[3] as usize;
        let data: Vec<f32> = out
            .to_device(tch::Device::Cpu)
            .contiguous()
            .try_into()
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Tensor::new(vec![1, 3, oh, ow], data))
    }
}

pub struct NcnnEngine;

impl NcnnEngine {
    pub fn new() -> Self {
        Self
    }
}

impl InferenceEngine for NcnnEngine {
    fn name(&self) -> &'static str {
        "ncnn"
    }

    fn capabilities(&self) -> EngineCaps {
        EngineCaps {
            backend: Backend::Vulkan,
            half: true,
            tiles: true,
        }
    }

    fn load(&mut self, _model: &ModelRef) -> Result<()> {
        Err(Error::unimplemented("NcnnEngine::load"))
    }

    fn infer(&mut self, _input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        Err(Error::unimplemented("NcnnEngine::infer"))
    }
}
