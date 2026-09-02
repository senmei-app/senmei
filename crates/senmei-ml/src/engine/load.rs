//! Architecture loading: build a `Model<B>` from a burnpack store.

use super::core::Model;
use crate::arch::{
    DisNet, Dncnn, Drunet, Ffdnet, IfrNet, NafNet, ParagonSrNet, RealPlk, RrdbNet, SafmnNet,
    Scunet, Span, SrvggNet, UpCunet2x, UpCunet2xFast,
};
use crate::model::ModelRef;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn_store::{BurnpackStore, ModuleSnapshot};

/// Build the arch on `device` from a burnpack `store` (f16 weights). The
/// 13-branch dispatch is identical for both engines; only the store's
/// from-adapter differs (tch converts f16→f32 at load).
pub fn load_arch<B: Backend>(
    model: &ModelRef,
    store: &mut BurnpackStore,
    device: &B::Device,
) -> Result<Model<B>> {
    match model.arch.as_str() {
        "upcunet2x" => {
            let mut m = UpCunet2x::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::UpCunet2x(m))
        }
        "upcunet2x-fast" | "fallin-cugan" => {
            // Fallin (renarchi CUGAN retrain) is an `UpCunet2x_fast` with
            // the same 38px reflect pad — only the weights differ.
            let mut m = UpCunet2xFast::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::UpCunet2xFast(m))
        }
        "realesrgan" => {
            let mut m = RrdbNet::new(
                model.scale as usize,
                model.num_block as usize,
                model.shuffle as usize,
                device,
            );
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::RrdbNet(m))
        }
        "srvgg" => {
            // Registered SRVGGNetCompact models: 64 features, body conv count
            // from the registry (16 animevideo-xs, 32 general-x4v3).
            let mut m = SrvggNet::new(64, model.num_conv as usize, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::SrvggNet(m))
        }
        "dis" => {
            // Registered DIS models: 32 features, body blocks from the
            // registry (8 DIS_Fast, 12 DIS_Balanced).
            let mut m = DisNet::new(32, model.num_block as usize, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::DisNet(m))
        }
        "ifrnet" => {
            let mut m = IfrNet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::IfrNet(m))
        }
        "drunet" => {
            let mut m = Drunet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Drunet(m))
        }
        "dncnn" => {
            let mut m = Dncnn::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Dncnn(m))
        }
        "ffdnet" => {
            let mut m = Ffdnet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Ffdnet(m))
        }
        "scunet" => {
            let mut m = Scunet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::Scunet(m))
        }
        "nafnet" => {
            let mut m = NafNet::new(device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::NafNet(m))
        }
        "real-plksr" => {
            let mut m = RealPlk::new(
                model.scale as usize,
                model.layer_norm,
                model.dysample,
                device,
            );
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::RealPlk(m))
        }
        "span" => {
            let mut m = Span::new(
                model.feature_channels as usize,
                model.scale as usize,
                device,
            );
            m.set_no_norm(model.no_norm);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            m.pad_k96(device);
            Ok(Model::Span(m))
        }
        "safmn" => {
            // SAFMN-L Real (registered models are fixed): dim 128 / 16 blocks
            // / ffn_scale 2.0; only the scale differs between x2 and x4.
            let mut m = SafmnNet::new(128, 16, 2.0, model.scale as usize, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::SafmnNet(m))
        }
        "paragonsr" => {
            // ParagonSR Nano (registered model is fixed): num_feat 24 / 3
            // residual groups × 2 blocks / ffn_expansion 1.5.
            let mut m = ParagonSrNet::new(model.scale as usize, 24, 3, 2, 1.5, device);
            m.load_from(store).map_err(|e| Error::new(e.to_string()))?;
            Ok(Model::ParagonSrNet(m))
        }
        other => Err(Error::new(format!("unsupported arch: {other}"))),
    }
}
