//! Backend-generic model architectures, shared by the `burn` (Vulkan) and
//! `tch` (libtorch) engines. Each arch is a clean re-implementation (see
//! per-module doc comments) and stays engine-agnostic: it only depends on
//! `burn::module` / `burn::tensor` types over `B: Backend`.

pub mod dncnn;
pub mod drunet;
pub mod ffdnet;
pub mod ifrnet;
pub mod nafnet;
pub mod paragonsr;
pub mod real_plksr;
pub mod realesrgan;
pub mod rife;
pub mod safmn;
pub mod scunet;
pub mod span;
pub mod srvgg;
pub mod upcunet;
pub mod warp;

pub use dncnn::Dncnn;
pub use drunet::Drunet;
pub use ffdnet::Ffdnet;
pub use ifrnet::IfrNet;
pub use nafnet::NafNet;
pub use paragonsr::ParagonSrNet;
pub use real_plksr::RealPlk;
pub use realesrgan::RrdbNet;
pub use rife::RifeNet;
pub use safmn::SafmnNet;
pub use scunet::Scunet;
pub use span::Span;
pub use srvgg::SrvggNet;
pub use upcunet::{UpCunet2x, UpCunet2xFast};
