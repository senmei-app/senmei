//! `.pth` → f16 `.bpk` conversion: per-arch key remapping + load/save.

use super::{ConvertOptions, ToF16};
use crate::arch::{
    Dncnn, Drunet, Ffdnet, IfrNet, NafNet, RealPlk, RrdbNet, SafmnNet, Scunet, Span, SrvggNet,
    UpCunet2x, UpCunet2xFast,
};
use crate::BurnBackend;
use crate::{Error, Result};
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
use burn_wgpu::WgpuDevice;

/// Maintainer step: f32 `.pth` → f16 `.bpk` via [`ToF16`].
pub fn convert_pth_to_bpk(opts: &ConvertOptions) -> Result<()> {
    let ConvertOptions {
        arch,
        pth_path,
        bpk_path,
        scale,
        num_block,
        layer_norm,
        dysample,
        shuffle,
    } = *opts;
    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(ToF16);
    match arch {
        "upcunet2x" | "upcunet2x-fast" | "fallin-cugan" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"\.conv\.0\.", ".conv.")
                .with_key_remapping(r"\.conv\.2\.", ".conv2.");
            match arch {
                "upcunet2x" => {
                    let mut m = UpCunet2x::<BurnBackend>::new(&device);
                    m.load_from(&mut store)
                        .map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save)
                        .map_err(|e| Error::new(e.to_string()))?;
                }
                _ => {
                    let mut m = UpCunet2xFast::<BurnBackend>::new(&device);
                    m.load_from(&mut store)
                        .map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save)
                        .map_err(|e| Error::new(e.to_string()))?;
                }
            }
        }
        "srvgg" => {
            let mut store = PytorchStore::from_file(pth_path);
            for (from, to) in srvgg_remap_patterns(num_block as usize) {
                store = store.with_key_remapping(from, to);
            }
            let mut m =
                SrvggNet::<BurnBackend>::new(64, num_block as usize, scale as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "realesrgan" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(
                    r"^RRDB_trunk\.(\d+)\.RDB(\d+)\.conv(\d+)\.",
                    "body.$1.rdb$2.conv$3.",
                )
                .with_key_remapping(r"^params_ema\.", "")
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"^trunk_conv\.", "conv_body.")
                .with_key_remapping(r"^upconv1\.", "conv_up1.")
                .with_key_remapping(r"^upconv2\.", "conv_up2.")
                .with_key_remapping(r"^HRconv\.", "conv_hr.");
            let mut m = RrdbNet::<BurnBackend>::new(
                scale as usize,
                num_block as usize,
                shuffle as usize,
                &device,
            );
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ifrnet" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"encoder\.pyramid(\d)\.(\d)\.0\.", "encoder.p$1.c$2.conv.")
                .with_key_remapping(r"encoder\.pyramid(\d)\.(\d)\.1\.", "encoder.p$1.c$2.prelu.")
                .with_key_remapping(r"decoder(\d)\.convblock\.0\.0\.", "decoder$1.cb0.conv.")
                .with_key_remapping(r"decoder(\d)\.convblock\.0\.1\.", "decoder$1.cb0.prelu.")
                .with_key_remapping(
                    r"decoder(\d)\.convblock\.1\.conv([1-4])\.0\.",
                    "decoder$1.cb1.c$2.conv.",
                )
                .with_key_remapping(
                    r"decoder(\d)\.convblock\.1\.conv([1-4])\.1\.",
                    "decoder$1.cb1.c$2.prelu.",
                )
                .with_key_remapping(r"decoder(\d)\.convblock\.1\.conv5\.", "decoder$1.cb1.c5.")
                .with_key_remapping(r"decoder(\d)\.convblock\.1\.prelu\.", "decoder$1.cb1.pl.")
                .with_key_remapping(r"decoder(\d)\.convblock\.2\.", "decoder$1.cb2.");
            let mut m = IfrNet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "drunet" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"m_down(\d)\.(\d)\.res\.0\.", "m_down$1.b$2.c1.")
                .with_key_remapping(r"m_down(\d)\.(\d)\.res\.2\.", "m_down$1.b$2.c2.")
                .with_key_remapping(r"m_down(\d)\.4\.", "m_down$1.down.")
                .with_key_remapping(r"m_body\.(\d)\.res\.0\.", "m_body.b$1.c1.")
                .with_key_remapping(r"m_body\.(\d)\.res\.2\.", "m_body.b$1.c2.")
                .with_key_remapping(r"m_up(\d)\.(\d)\.res\.0\.", "m_up$1.b$2.c1.")
                .with_key_remapping(r"m_up(\d)\.(\d)\.res\.2\.", "m_up$1.b$2.c2.")
                .with_key_remapping(r"m_up(\d)\.0\.", "m_up$1.up.");
            let mut m = Drunet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "dncnn" => {
            let mut store =
                PytorchStore::from_file(pth_path).with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Dncnn::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ffdnet" => {
            let mut store =
                PytorchStore::from_file(pth_path).with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Ffdnet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "scunet" => {
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(ToF16);
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^module\.", "")
                .with_key_remapping(r"^m_head\.0\.", "m_head.")
                .with_key_remapping(r"^m_tail\.0\.", "m_tail.")
                .with_key_remapping(r"^m_down(\d)\.4\.", "m_down${1}_down.")
                .with_key_remapping(r"^m_up(\d)\.0\.", "m_up${1}_up.")
                .with_key_remapping(r"\.trans_block\.mlp\.0\.", ".trans_block.mlp0.")
                .with_key_remapping(r"\.trans_block\.mlp\.2\.", ".trans_block.mlp2.")
                .with_key_remapping(r"\.conv_block\.0\.", ".conv_block.c0.")
                .with_key_remapping(r"\.conv_block\.2\.", ".conv_block.c2.")
                .with_key_remapping(r"\.ln([12])\.weight", ".ln$1.gamma")
                .with_key_remapping(r"\.ln([12])\.bias", ".ln$1.beta");
            let mut m = Scunet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "nafnet" => {
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(ToF16);
            let mut store = PytorchStore::from_file(pth_path)
                .with_top_level_key("params")
                .with_key_remapping(r"^encoders\.(\d+)\.(\d+)\.", "encoders.$1.blocks.$2.")
                .with_key_remapping(r"^decoders\.(\d+)\.(\d+)\.", "decoders.$1.blocks.$2.")
                .with_key_remapping(r"^middle_blks\.(\d+)\.", "middle.$1.")
                .with_key_remapping(r"^ups\.(\d+)\.0\.", "ups.$1.conv.")
                .with_key_remapping(r"sca\.1\.", "sca_conv.");
            let mut m = NafNet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "real-plksr" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"^feats\.0\.", "head.")
                .with_key_remapping(r"^feats\.30\.", "tail.")
                .with_key_remapping(r"^to_img\.", "")
                .with_key_remapping(r"\.channel_mixer\.0\.", ".channel_mixer.conv1.")
                .with_key_remapping(r"\.channel_mixer\.2\.", ".channel_mixer.conv2.")
                .with_key_remapping(r"\.attn\.f\.0\.", ".attn.f.");
            if layer_norm {
                store = store.with_key_remapping(r"\.norm\.", ".layer_norm.");
            }
            let store = (1..=28usize).fold(store, |s, i| {
                s.with_key_remapping(format!(r"^feats\.{i}\."), format!("blocks.{}.", i - 1))
            });
            let mut store = store;
            let mut m = RealPlk::<BurnBackend>::new(scale as usize, layer_norm, dysample, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "span" => {
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^params\.", "")
                .with_key_remapping(r"\.conv\.0\.", ".conv0.")
                .with_key_remapping(r"\.conv\.1\.", ".conv1.")
                .with_key_remapping(r"\.conv\.2\.", ".conv2.")
                .with_key_remapping(r"^upsampler\.0\.", "upsampler.");
            let mut m = Span::<BurnBackend>::new(num_block as usize, scale as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "safmn" => {
            let mut store = PytorchStore::from_file(pth_path);
            for (from, to) in safmn_remap_patterns() {
                store = store.with_key_remapping(from, to);
            }
            let mut m =
                SafmnNet::<BurnBackend>::new(128, num_block as usize, 2.0, scale as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Remap patterns (shared by converter + tests)
// ---------------------------------------------------------------------------

/// Remap rules for the SRVGG checkpoints. `num_conv` is the mid-conv count
/// (16 animevideo-xs, 32 general-x4v3).
pub(crate) fn srvgg_remap_patterns(num_conv: usize) -> Vec<(String, String)> {
    let mut patterns = vec![(r"^params\.".to_string(), String::new())];
    for k in 0..=num_conv {
        patterns.push((
            format!(r"^body\.{}\.weight$", k * 2 + 1),
            format!("prelu.{k}.weight"),
        ));
    }
    for i in 0..num_conv + 2 {
        patterns.push((
            format!(r"body\.{}\.(weight|bias)", i * 2),
            format!("body.{}.$1", i),
        ));
    }
    patterns
}

/// Remap rules for the SAFMN checkpoints.
pub(crate) fn safmn_remap_patterns() -> Vec<(String, String)> {
    vec![
        (r"^params_ema\.".to_string(), String::new()),
        (r"^params\.".to_string(), String::new()),
        (r"^feats\.(\d+)\.".to_string(), "blocks.$1.".into()),
        (r"\.ccm\.ccm\.0\.".to_string(), ".ccm.conv1.".into()),
        (r"\.ccm\.ccm\.2\.".to_string(), ".ccm.conv2.".into()),
        (r"^to_img\.0\.".to_string(), "to_img_conv.".into()),
    ]
}

/// Remap rules for the DIS (scale-2) checkpoints.
pub(crate) fn dis_remap_patterns() -> Vec<(String, String)> {
    vec![(
        r"^upsampler\.(conv|act)\.".to_string(),
        "upsampler.0.$1.".to_string(),
    )]
}
