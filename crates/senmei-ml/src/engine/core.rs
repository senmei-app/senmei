//! Backend-generic engine core, shared by the burn (Vulkan f16) and tch
//! (libtorch f32) engines. Holds the arch `Model<B>` enum plus the basic
//! load/infer logic; only the element cast (`B::FloatElem`) and `B::Device`
//! differ, both passed in by the engines.
#![cfg(any(feature = "burn", feature = "tch"))]

use crate::arch::{
    DisNet, Dncnn, Drunet, Ffdnet, IfrNet, NafNet, ParagonSrNet, RealPlk, RifeNet, RrdbNet,
    SafmnNet, Scunet, Span, SrvggNet, UpCunet2x, UpCunet2xFast,
};
use crate::tensor::Tensor;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor as BurnTensor, TensorData};

/// The loaded arch, generic over the backend (`BurnBackend<f16>` or
/// `LibTorch<f32>`).
pub enum Model<B: Backend> {
    UpCunet2x(UpCunet2x<B>),
    UpCunet2xFast(UpCunet2xFast<B>),
    RrdbNet(RrdbNet<B>),
    SrvggNet(SrvggNet<B>),
    RifeNet(Box<RifeNet<B>>),
    IfrNet(IfrNet<B>),
    Drunet(Drunet<B>),
    Dncnn(Dncnn<B>),
    Ffdnet(Ffdnet<B>),
    Scunet(Scunet<B>),
    NafNet(NafNet<B>),
    RealPlk(RealPlk<B>),
    Span(Span<B>),
    SafmnNet(SafmnNet<B>),
    ParagonSrNet(ParagonSrNet<B>),
    DisNet(DisNet<B>),
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: BurnTensor<B, 4>) -> Result<BurnTensor<B, 4>> {
        match self {
            Model::UpCunet2x(m) => Ok(m.forward(x)),
            Model::UpCunet2xFast(m) => Ok(m.forward(x)),
            Model::RrdbNet(m) => Ok(m.forward(x)),
            Model::SrvggNet(m) => Ok(m.forward(x)),
            Model::RealPlk(m) => Ok(m.forward(x)),
            Model::Drunet(m) => Ok(m.forward(x)),
            Model::Dncnn(m) => Ok(m.forward(x)),
            Model::Scunet(m) => Ok(m.forward(x)),
            Model::NafNet(m) => Ok(m.forward(x)),
            Model::Span(m) => Ok(m.forward(x)),
            Model::SafmnNet(m) => Ok(m.forward(x)),
            Model::ParagonSrNet(m) => Ok(m.forward(x)),
            Model::DisNet(m) => Ok(m.forward(x)),
            Model::RifeNet(_) | Model::IfrNet(_) | Model::Ffdnet(_) => {
                Err(Error::new("no single-input forward"))
            }
        }
    }

    pub fn interp(
        &self,
        a: BurnTensor<B, 4>,
        b: BurnTensor<B, 4>,
        t: BurnTensor<B, 4>,
    ) -> Result<BurnTensor<B, 4>> {
        match self {
            Model::RifeNet(m) => Ok(m.forward(a, b, t)),
            Model::IfrNet(m) => Ok(m.forward(a, b, t)),
            _ => Err(Error::new("model has no frame interpolation")),
        }
    }

    pub fn is_rife(&self) -> bool {
        matches!(self, Model::RifeNet(_))
    }
}

fn to_burn<B: Backend>(input: &Tensor, device: &B::Device) -> Result<BurnTensor<B, 4>> {
    if input.shape.len() != 4 {
        return Err(Error::new("expected NCHW input"));
    }
    let [n, c, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
    Ok(BurnTensor::<B, 4>::from_data(
        TensorData::new(input.data.clone(), [n, c, h, w]).convert::<B::FloatElem>(),
        device,
    ))
}

fn to_tensor<B: Backend>(out: BurnTensor<B, 4>, shape: [usize; 4]) -> Result<Tensor> {
    let data = out
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| Error::new(e.to_string()))?;
    Ok(Tensor::new(shape.to_vec(), data))
}

/// Models whose single-input `forward` takes a 3-channel RGB tensor (used to
/// pick warmup inputs; DRUNet wants 4ch, FFDNet/RIFE/IFRNet have no
/// single-input forward at all).
#[cfg(feature = "burn")]
pub fn single_input_rgb<B: Backend>(model: &Model<B>) -> bool {
    !matches!(
        model,
        Model::Drunet(_) | Model::Ffdnet(_) | Model::RifeNet(_) | Model::IfrNet(_)
    )
}

pub fn infer<B: Backend>(model: &Model<B>, input: &Tensor, device: &B::Device) -> Result<Tensor> {
    let x = to_burn::<B>(input, device)?;
    let out = model.forward(x)?;
    let [_, _, oh, ow] = out.dims();
    to_tensor(out, [input.shape[0], input.shape[1], oh, ow])
}

pub fn infer_interp<B: Backend>(
    model: &Model<B>,
    a: &Tensor,
    b: &Tensor,
    t: f32,
    device: &B::Device,
) -> Option<Result<Tensor>> {
    if !matches!(model, Model::RifeNet(_) | Model::IfrNet(_)) {
        return None; // not an interpolation model → caller falls back
    }
    let [n, c, h, w] = [a.shape[0], a.shape[1], a.shape[2], a.shape[3]];
    let a_t = match to_burn::<B>(a, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let b_t = match to_burn::<B>(b, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    // The flow estimators run on a downscaled grid (RIFE 1/32, IFRNet 1/16
    // via its pyramid), so pad to a multiple and crop back (like the refs).
    let pad = if model.is_rife() { 32 } else { 16 };
    let pad_h = h.div_ceil(pad) * pad;
    let pad_w = w.div_ceil(pad) * pad;
    let pad = |x: BurnTensor<B, 4>| {
        let mut x = x;
        if pad_h > h {
            let z = BurnTensor::<B, 4>::zeros([n, c, pad_h - h, w], device);
            x = BurnTensor::cat(vec![x, z], 2);
        }
        if pad_w > w {
            let z = BurnTensor::<B, 4>::zeros([n, c, pad_h, pad_w - w], device);
            x = BurnTensor::cat(vec![x, z], 3);
        }
        x
    };
    let a_t = pad(a_t);
    let b_t = pad(b_t);
    // ncnn broadcasts the scalar timestep over the (padded) spatial grid.
    let t_t = BurnTensor::<B, 4>::ones([n, 1, pad_h, pad_w], device) * t;
    let out = match model.interp(a_t, b_t, t_t) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    let out = out.slice([0..n, 0..c, 0..h, 0..w]);
    Some(to_tensor(out, [n, c, h, w]))
}

/// Denoise dispatch: DRUNet needs 4ch (3+sigma) + 8-aligned padding,
/// FFDNet takes sigma internally, DnCNN/SCUNet are blind. `None` = no denoise.
pub fn infer_denoise<B: Backend>(
    model: &Model<B>,
    input: &Tensor,
    sigma: f32,
    device: &B::Device,
) -> Option<Result<Tensor>> {
    let is_drunet = matches!(model, Model::Drunet(_));
    if !matches!(
        model,
        Model::Drunet(_) | Model::Dncnn(_) | Model::Ffdnet(_) | Model::Scunet(_)
    ) {
        return None;
    }
    if input.shape.len() != 4 || input.shape[1] != 3 {
        return Some(Err(Error::new("expected 3-channel NCHW input")));
    }
    let [n, _c, h, w] = [
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    ];
    let rgb = match to_burn::<B>(input, device) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    if let Model::Ffdnet(m) = model {
        let out = m.forward(rgb, sigma);
        return Some(to_tensor(out, [n, 3, h, w]));
    }
    if !is_drunet {
        let out = match model.forward(rgb) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        return Some(to_tensor(out, [n, 3, h, w]));
    }
    let sigma_map = BurnTensor::<B, 4>::ones([n, 1, h, w], device) * sigma;
    let x = BurnTensor::cat(vec![rgb, sigma_map], 1);
    let pad_h = h.div_ceil(8) * 8;
    let pad_w = w.div_ceil(8) * 8;
    let mut x = x;
    if pad_h > h {
        let z = BurnTensor::<B, 4>::zeros([n, 4, pad_h - h, w], device);
        x = BurnTensor::cat(vec![x, z], 2);
    }
    if pad_w > w {
        let z = BurnTensor::<B, 4>::zeros([n, 4, pad_h, pad_w - w], device);
        x = BurnTensor::cat(vec![x, z], 3);
    }
    let out = match model.forward(x) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    let out = out.slice([0..n, 0..3, 0..h, 0..w]);
    Some(to_tensor(out, [n, 3, h, w]))
}
