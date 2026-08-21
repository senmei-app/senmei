//! Backend-generic model architectures, shared by the `burn` (Vulkan) and
//! `tch` (libtorch) engines. Each arch is a clean re-implementation (see
//! per-module doc comments) and stays engine-agnostic: it only depends on
//! `burn::module` / `burn::tensor` types over `B: Backend`.

pub mod realesrgan;
pub mod rife;
pub mod srvgg;
pub mod upcunet;
pub mod warp;

pub use realesrgan::RrdbNet;
pub use rife::RifeNet;
pub use srvgg::SrvggNet;
pub use upcunet::{UpCunet2x, UpCunet2xFast};
