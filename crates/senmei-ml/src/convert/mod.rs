//! One-time `.pth`/`.onnx`/safetensors → f16 `.bpk` conversion for the burn
//! engine (maintainer + `download_model`). Loads the f32 state dict on the
//! Vulkan backend and saves through [`ToF16`] so `BurnEngine` can load it as f16.

mod onnx;
mod pth;
mod safetensors;

use burn::tensor::DType;
use burn_store::{ModuleAdapter, TensorSnapshot};
use std::path::Path;

pub use onnx::convert_onnx_to_bpk;
pub use pth::convert_pth_to_bpk;
pub use safetensors::convert_safetensors_to_bpk;

/// Cast every stored F32 tensor to F16 — the conversion's goal is an all-f16
/// burnpack. Casting unconditionally is safe: none of the archs use BatchNorm
/// (whose `running_var` underflows in f16).
#[derive(Clone)]
struct ToF16;

impl ModuleAdapter for ToF16 {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let target = match snapshot.dtype {
            DType::F32 => DType::F16,
            _ => return snapshot.clone(),
        };
        let original = snapshot.clone_data_fn();
        let cast = std::rc::Rc::new(move || Ok(original()?.convert_dtype(target)));
        TensorSnapshot::from_closure(
            cast,
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Conversion knobs for the `.pth` → `.bpk` maintainer tool.
#[derive(Clone, Copy)]
pub struct ConvertOptions<'a> {
    pub arch: &'a str,
    pub pth_path: &'a Path,
    pub bpk_path: &'a Path,
    pub scale: u32,
    pub num_block: u32,
    pub layer_norm: bool,
    pub dysample: bool,
    pub shuffle: u32,
}

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::pth::{dis_remap_patterns, safmn_remap_patterns, srvgg_remap_patterns};
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use burn_store::{KeyRemapper, TensorSnapshot};

    #[test]
    fn safmn_conversion_key_contract() {
        let mut source = Vec::with_capacity(292);
        source.push("params_ema.to_feat.weight".into());
        source.push("params_ema.to_feat.bias".into());
        for i in 0..16u32 {
            source.push(format!("params_ema.feats.{i}.norm1.weight"));
            source.push(format!("params_ema.feats.{i}.norm1.bias"));
            source.push(format!("params_ema.feats.{i}.norm2.weight"));
            source.push(format!("params_ema.feats.{i}.norm2.bias"));
            for j in 0..4u32 {
                source.push(format!("params_ema.feats.{i}.safm.mfr.{j}.weight"));
                source.push(format!("params_ema.feats.{i}.safm.mfr.{j}.bias"));
            }
            source.push(format!("params_ema.feats.{i}.safm.aggr.weight"));
            source.push(format!("params_ema.feats.{i}.safm.aggr.bias"));
            source.push(format!("params_ema.feats.{i}.ccm.ccm.0.weight"));
            source.push(format!("params_ema.feats.{i}.ccm.ccm.0.bias"));
            source.push(format!("params_ema.feats.{i}.ccm.ccm.2.weight"));
            source.push(format!("params_ema.feats.{i}.ccm.ccm.2.bias"));
        }
        source.push("params_ema.to_img.0.weight".into());
        source.push("params_ema.to_img.0.bias".into());
        assert_eq!(source.len(), 292);

        let snapshots = source
            .iter()
            .map(|name| {
                let mut s = TensorSnapshot::from_data(
                    TensorData::new(vec![0f32; 1], vec![1]),
                    name.split('.').map(str::to_string).collect(),
                    Vec::new(),
                    ParamId::new(),
                );
                s.container_stack = None;
                s.tensor_id = None;
                s
            })
            .collect();

        let remapper = KeyRemapper::from_patterns(safmn_remap_patterns()).unwrap();
        let (remapped, _) = remapper.remap(snapshots);
        let mut paths: Vec<String> = remapped.iter().map(|s| s.full_path()).collect();
        paths.sort();
        paths.dedup();

        let mut expected = Vec::with_capacity(292);
        expected.push("to_feat.weight".into());
        expected.push("to_feat.bias".into());
        for i in 0..16u32 {
            expected.push(format!("blocks.{i}.norm1.weight"));
            expected.push(format!("blocks.{i}.norm1.bias"));
            expected.push(format!("blocks.{i}.norm2.weight"));
            expected.push(format!("blocks.{i}.norm2.bias"));
            for j in 0..4u32 {
                expected.push(format!("blocks.{i}.safm.mfr.{j}.weight"));
                expected.push(format!("blocks.{i}.safm.mfr.{j}.bias"));
            }
            expected.push(format!("blocks.{i}.safm.aggr.weight"));
            expected.push(format!("blocks.{i}.safm.aggr.bias"));
            expected.push(format!("blocks.{i}.ccm.conv1.weight"));
            expected.push(format!("blocks.{i}.ccm.conv1.bias"));
            expected.push(format!("blocks.{i}.ccm.conv2.weight"));
            expected.push(format!("blocks.{i}.ccm.conv2.bias"));
        }
        expected.push("to_img_conv.weight".into());
        expected.push("to_img_conv.bias".into());
        expected.sort();

        assert_eq!(paths, expected);
    }

    #[test]
    fn srvgg_conversion_key_contract() {
        let num_conv = 16usize;
        let mut source = Vec::with_capacity(2 * (num_conv + 2) + num_conv + 1);
        for i in 0..num_conv + 2 {
            source.push(format!("params.body.{}.weight", i * 2));
            source.push(format!("params.body.{}.bias", i * 2));
        }
        for k in 0..=num_conv {
            source.push(format!("params.body.{}.weight", k * 2 + 1));
        }

        let snapshots = source
            .iter()
            .map(|name| {
                let mut s = TensorSnapshot::from_data(
                    TensorData::new(vec![0f32; 1], vec![1]),
                    name.split('.').map(str::to_string).collect(),
                    Vec::new(),
                    ParamId::new(),
                );
                s.container_stack = None;
                s.tensor_id = None;
                s
            })
            .collect();

        let remapper = KeyRemapper::from_patterns(srvgg_remap_patterns(num_conv)).unwrap();
        let (remapped, _) = remapper.remap(snapshots);
        let mut paths: Vec<String> = remapped.iter().map(|s| s.full_path()).collect();
        paths.sort();
        paths.dedup();

        let mut expected = Vec::with_capacity(paths.len());
        for i in 0..num_conv + 2 {
            expected.push(format!("body.{}.weight", i));
            expected.push(format!("body.{}.bias", i));
        }
        for k in 0..=num_conv {
            expected.push(format!("prelu.{k}.weight"));
        }
        expected.sort();

        assert_eq!(paths, expected);
    }

    #[test]
    fn dis_conversion_key_contract() {
        let num_blocks = 8usize;
        let mut source = Vec::new();
        source.push("head.weight".into());
        source.push("head.bias".into());
        source.push("head_act.weight".into());
        for i in 0..num_blocks {
            source.push(format!("body.{i}.conv1.weight"));
            source.push(format!("body.{i}.conv2.weight"));
            source.push(format!("body.{i}.act.weight"));
        }
        source.push("fusion.weight".into());
        source.push("fusion.bias".into());
        source.push("upsampler.conv.weight".into());
        source.push("upsampler.conv.bias".into());
        source.push("upsampler.act.weight".into());
        source.push("tail.weight".into());
        source.push("tail.bias".into());

        let snapshots = source
            .iter()
            .map(|name| {
                let mut s = TensorSnapshot::from_data(
                    TensorData::new(vec![0f32; 1], vec![1]),
                    name.split('.').map(str::to_string).collect(),
                    Vec::new(),
                    ParamId::new(),
                );
                s.container_stack = None;
                s.tensor_id = None;
                s
            })
            .collect();

        let remapper = KeyRemapper::from_patterns(dis_remap_patterns()).unwrap();
        let (remapped, _) = remapper.remap(snapshots);
        let mut paths: Vec<String> = remapped.iter().map(|s| s.full_path()).collect();
        paths.sort();
        paths.dedup();

        let mut expected = Vec::new();
        expected.push("head.weight".into());
        expected.push("head.bias".into());
        expected.push("head_act.weight".into());
        for i in 0..num_blocks {
            expected.push(format!("body.{i}.conv1.weight"));
            expected.push(format!("body.{i}.conv2.weight"));
            expected.push(format!("body.{i}.act.weight"));
        }
        expected.push("fusion.weight".into());
        expected.push("fusion.bias".into());
        expected.push("upsampler.0.conv.weight".into());
        expected.push("upsampler.0.conv.bias".into());
        expected.push("upsampler.0.act.weight".into());
        expected.push("tail.weight".into());
        expected.push("tail.bias".into());
        expected.sort();

        assert_eq!(paths, expected);
    }
}
