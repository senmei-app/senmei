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

pub struct TorchEngine;

impl TorchEngine {
    pub fn new() -> Self {
        Self
    }
}

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
        Err(Error::unimplemented("TorchEngine::load"))
    }

    fn infer(&mut self, _input: &Tensor, _opts: &InferOptions) -> Result<Tensor> {
        Err(Error::unimplemented("TorchEngine::infer"))
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
