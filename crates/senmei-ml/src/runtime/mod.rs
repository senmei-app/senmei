//! Runtime dependency resolution: hardware detection and (later) on-demand
//! libtorch download/loading — the desktop app resolves libtorch at runtime
//! (CUDA/ROCm only, no CPU), like Koharu, instead of a build-time link.

pub mod hardware;
pub mod rocm;
pub mod torch;

pub use hardware::{detect, vram_available_bytes, vram_total_bytes, Hardware};
pub use torch::{pick_device, pick_variant, resolve, TorchInstall, TorchVariant};
