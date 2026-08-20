//! DnCNN (cszn/KAIR, MIT) — clean burn port.
//!
//! 20 conv layers (3→64, 18× 64→64, 64→3) with ReLU between, no BatchNorm
//! (the `dncnn_color_blind` checkpoint trains with `act_mode="R"`). Residual
//! learning: `out = x - model(x)` — the net predicts the additive noise map.
//! All stride-1 convs, so any input size works without padding to a multiple.
//!
//! Field names carry the torch `model.{2i}` indices (the ReLU layers sit at
//! odd `model.{2i+1}` slots and carry no params), so the converter only needs
//! `^model\.(\d+)\.` → `c$1.` — no renumbering.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

fn conv3<B: Backend>(in_c: usize, out_c: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([in_c, out_c], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

#[derive(Module, Debug)]
pub struct Dncnn<B: Backend> {
    c0: Conv2d<B>,
    c2: Conv2d<B>,
    c4: Conv2d<B>,
    c6: Conv2d<B>,
    c8: Conv2d<B>,
    c10: Conv2d<B>,
    c12: Conv2d<B>,
    c14: Conv2d<B>,
    c16: Conv2d<B>,
    c18: Conv2d<B>,
    c20: Conv2d<B>,
    c22: Conv2d<B>,
    c24: Conv2d<B>,
    c26: Conv2d<B>,
    c28: Conv2d<B>,
    c30: Conv2d<B>,
    c32: Conv2d<B>,
    c34: Conv2d<B>,
    c36: Conv2d<B>,
    c38: Conv2d<B>,
}

impl<B: Backend> Dncnn<B> {
    pub fn new(device: &B::Device) -> Self {
        let c = |i: usize, o: usize| conv3(i, o, device);
        Self {
            c0: c(3, 64),
            c2: c(64, 64),
            c4: c(64, 64),
            c6: c(64, 64),
            c8: c(64, 64),
            c10: c(64, 64),
            c12: c(64, 64),
            c14: c(64, 64),
            c16: c(64, 64),
            c18: c(64, 64),
            c20: c(64, 64),
            c22: c(64, 64),
            c24: c(64, 64),
            c26: c(64, 64),
            c28: c(64, 64),
            c30: c(64, 64),
            c32: c(64, 64),
            c34: c(64, 64),
            c36: c(64, 64),
            c38: c(64, 3),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = relu(self.c0.forward(x.clone()));
        h = relu(self.c2.forward(h));
        h = relu(self.c4.forward(h));
        h = relu(self.c6.forward(h));
        h = relu(self.c8.forward(h));
        h = relu(self.c10.forward(h));
        h = relu(self.c12.forward(h));
        h = relu(self.c14.forward(h));
        h = relu(self.c16.forward(h));
        h = relu(self.c18.forward(h));
        h = relu(self.c20.forward(h));
        h = relu(self.c22.forward(h));
        h = relu(self.c24.forward(h));
        h = relu(self.c26.forward(h));
        h = relu(self.c28.forward(h));
        h = relu(self.c30.forward(h));
        h = relu(self.c32.forward(h));
        h = relu(self.c34.forward(h));
        h = relu(self.c36.forward(h));
        x - self.c38.forward(h)
    }
}
