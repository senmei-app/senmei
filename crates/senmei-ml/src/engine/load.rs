//! Architecture loading: build a `Model<B>` from a burnpack store.

use super::core::Model;
use crate::arch::{
    DisNet, Dncnn, Drunet, Ffdnet, IfrNet, NafNet, ParagonSrNet, RealPlk, RrdbNet, SafmnNet,
    Scunet, Span, SrvggNet, UpCunet2x, UpCunet2xFast,
};
use crate::model::ModelRef;
use crate::{Result, ResultExt};
use burn::tensor::backend::Backend;
use burn_store::{BurnpackStore, ModuleSnapshot};

/// Shared by both engines — only the store's from-adapter differs
/// (tch converts f16→f32 at load).
pub fn load_arch<B: Backend>(
    model: &ModelRef,
    store: &mut BurnpackStore,
    device: &B::Device,
) -> Result<Model<B>> {
    match model.arch.as_str() {
        "upcunet2x" => {
            let mut m = UpCunet2x::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::UpCunet2x(m))
        }
        "upcunet2x-fast" | "fallin-cugan" => {
            let mut m = UpCunet2xFast::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::UpCunet2xFast(m))
        }
        "realesrgan" => {
            let mut m = RrdbNet::new(
                model.scale as usize,
                model.num_block as usize,
                model.shuffle as usize,
                device,
            );
            m.load_from(store).map_err_str()?;
            Ok(Model::RrdbNet(m))
        }
        "srvgg" => {
            let mut m = SrvggNet::new(64, model.num_conv as usize, model.scale as usize, device);
            m.load_from(store).map_err_str()?;
            Ok(Model::SrvggNet(m))
        }
        "dis" => {
            let mut m = DisNet::new(32, model.num_block as usize, model.scale as usize, device);
            m.load_from(store).map_err_str()?;
            Ok(Model::DisNet(m))
        }
        "ifrnet" => {
            let mut m = IfrNet::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::IfrNet(m))
        }
        "drunet" => {
            let mut m = Drunet::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::Drunet(m))
        }
        "dncnn" => {
            let mut m = Dncnn::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::Dncnn(m))
        }
        "ffdnet" => {
            let mut m = Ffdnet::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::Ffdnet(m))
        }
        "scunet" => {
            let mut m = Scunet::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::Scunet(m))
        }
        "nafnet" => {
            let mut m = NafNet::new(device);
            m.load_from(store).map_err_str()?;
            Ok(Model::NafNet(m))
        }
        "real-plksr" => {
            let mut m = RealPlk::new(
                model.scale as usize,
                model.layer_norm,
                model.dysample,
                device,
            );
            m.load_from(store).map_err_str()?;
            Ok(Model::RealPlk(m))
        }
        "span" => {
            let mut m = Span::new(
                model.feature_channels as usize,
                model.scale as usize,
                device,
            );
            m.set_no_norm(model.no_norm);
            m.load_from(store).map_err_str()?;
            m.pad_k96(device);
            Ok(Model::Span(m))
        }
        "safmn" => {
            let mut m = SafmnNet::new(128, 16, 2.0, model.scale as usize, device);
            m.load_from(store).map_err_str()?;
            Ok(Model::SafmnNet(m))
        }
        "paragonsr" => {
            let mut m = ParagonSrNet::new(model.scale as usize, 24, 3, 2, 1.5, device);
            m.load_from(store).map_err_str()?;
            Ok(Model::ParagonSrNet(m))
        }
        other => Err(crate::Error::new(format!("unsupported arch: {other}"))),
    }
}
