//! One-time `.pth`/`.onnx` → f16 `.bpk` conversion for the burn engine
//! (maintainer + `download_model`). Loads the f32 state dict on the Vulkan
//! backend and saves through `HalfPrecisionAdapter` so `BurnEngine` can load
//! it as f16.

use crate::arch::{
    Dncnn, Drunet, Ffdnet, IfrNet, NafNet, RealPlk, RrdbNet, Scunet, Span, SrvggNet, UpCunet2x,
    UpCunet2xFast,
};
use crate::BurnBackend;
use crate::{Error, Result};
use burn::module::ParamId;
use burn::tensor::backend::Backend;
use burn::tensor::{f16, TensorData};
use burn_store::{
    BurnpackStore, HalfPrecisionAdapter, KeyRemapper, ModuleSnapshot, PytorchStore, TensorSnapshot,
};
use burn_wgpu::WgpuDevice;
use std::path::Path;

/// One-time `.pth` → f16 `.bpk` conversion for an arch (maintainer step).
/// Loads the f32 state dict on the Vulkan backend (upcunet key remap), then
/// saves through `HalfPrecisionAdapter` so `BurnEngine` can load it as f16.
pub fn convert_pth_to_bpk(
    arch: &str,
    pth_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
    layer_norm: bool,
    dysample: bool,
) -> Result<()> {
    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path)
        .with_to_adapter(HalfPrecisionAdapter::new().with_module("Prelu"));
    match arch {
        "upcunet2x" | "upcunet2x-fast" | "fallin-cugan" => {
            let mut store = PytorchStore::from_file(pth_path)
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
                    // upcunet2x-fast and fallin-cugan share the module layout.
                    let mut m = UpCunet2xFast::<BurnBackend>::new(&device);
                    m.load_from(&mut store)
                        .map_err(|e| Error::new(e.to_string()))?;
                    m.save_into(&mut save)
                        .map_err(|e| Error::new(e.to_string()))?;
                }
            }
        }
        "srvgg" => {
            // Torch SRVGG body is flat (`body.{0,2,4,…}` = convs, `body.{1,3,…}`
            // = the ONE shared PReLU). Remap the even conv indices onto the Vec;
            // `body.1.weight` feeds the shared PReLU and the duplicate odd
            // weights stay unused. 4× upsampler convs sit at `upsampler.0/.2`
            // (PixelShuffle between), so `.2` becomes the Vec's `.1`.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(
                    r"body\.(1|3|5|7|9|11|13|15|17|19|21|23|25|27|29)\.weight",
                    "prelu.weight",
                );
            for i in 0..16u32 {
                let from = format!(r"body\.{}\.(weight|bias)", i * 2);
                let to = format!("body.{}.$1", i);
                store = store.with_key_remapping(from, to);
            }
            store = store.with_key_remapping(r"upsampler\.2\.", "upsampler.1.");
            let mut m = SrvggNet::<BurnBackend>::new(64, 16, scale as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "realesrgan" => {
            // Also handles BSRGAN (KAIR): same RRDBNet, but its keys use the
            // older BasicSR naming (`RRDB_trunk.{i}.RDB{j}.conv{k}`, `trunk_conv`,
            // `upconv1/2`, `HRconv`); the rules only match those, so standard
            // Real-ESRGAN pths (`body.{i}.rdb{j}.conv{k}`, `conv_body`,
            // `conv_up1/2`, `conv_hr`) pass through unchanged.
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
            let mut m = RrdbNet::<BurnBackend>::new(scale as usize, num_block as usize, &device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ifrnet" => {
            // Torch Sequential/ResBlock keys (pyramid1.0.0, convblock.1.conv1.0,
            // …) are mapped onto the burn field paths (p1.c0.conv, cb1.c1.conv,
            // …) with capture-group rules; strips a DataParallel `module.` prefix.
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
            // Torch Sequential ResBlock keys (m_down1.0.res.0/.res.2, the
            // index-4 stride-conv m_down1.4, and the index-0 deconv m_up3.0)
            // are mapped onto the burn field paths (b0.c1/b0.c2, down, up)
            // with capture-group rules.
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
            // Torch `model.{2i}.weight/bias` (ReLU sits at odd `{2i+1}` slots,
            // no params) map onto the burn `c{2i}` field names 1:1.
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Dncnn::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "ffdnet" => {
            // Same `model.{2i}` layout as DnCNN (ReLU at odd slots).
            let mut store = PytorchStore::from_file(pth_path)
                .with_key_remapping(r"^model\.(\d+)\.", "c$1.");
            let mut m = Ffdnet::<BurnBackend>::new(&device);
            m.load_from(&mut store)
                .map_err(|e| Error::new(e.to_string()))?;
            m.save_into(&mut save)
                .map_err(|e| Error::new(e.to_string()))?;
        }
        "scunet" => {
            // Torch `m_{head,down,body,up,tail}` Sequential keys map onto the
            // burn field paths: head/tail are `m_head.0.`/`m_tail.0.`; down
            // levels keep block indices 0-3 and the index-4 stride conv maps
            // to `_down`; up levels map the index-0 deconv to `_up`. MLP/conv
            // blocks are torch Sequentials (`.mlp.0`/`.mlp.2`,
            // `.conv_block.0`/`.conv_block.2`) and LayerNorm weight/bias are
            // burn `gamma`/`beta`.
            //
            // The `relative_position_params` bare-tensor param lives in the
            // custom `Wmsa` module, which is not in the default half-precision
            // set — add it so the f16 bpk stores it as F16 (otherwise the f16
            // model loads it F32 and the attention add fails DTypeMismatch).
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(
                HalfPrecisionAdapter::new().with_module("Wmsa"),
            );
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
            // Torch NAFBlock keys (encoders.0.0.conv1, sca.1, middle_blks.0,
            // ups.0.0, downs.0) map onto the burn field paths
            // (encoders.0.blocks.0.conv1, sca_conv, middle.0, ups.0.conv,
            // downs.0) with capture-group rules. The checkpoint wraps the
            // state dict under `params`. The custom `NafBlock`/`LayerNorm2d`
            // structs hold `beta`/`gamma`/norm params that aren't in the
            // default half-precision set, so add them for the f16 conversion.
            let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(
                HalfPrecisionAdapter::new()
                    .with_module("NafBlock")
                    .with_module("LayerNorm2d"),
            );
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
            // Remap the torch `feats.{i}` / `to_img.` keys onto the module
            // record paths (`head`/`blocks`/`tail`, and `offset`/`scope`/
            // `end_conv`). The channel_mixer/attn are torch `nn.Sequential`,
            // so their sub-convs are indexed (`channel_mixer.0`/`.2`,
            // `attn.f.0`) rather than named. LayerNorm blocks keep the torch
            // `feats.{i}.norm.{weight,bias}` name (per-pixel channel norm →
            // record `blocks.{i-1}.layer_norm.{weight,bias}`), so remap
            // `norm.` → `layer_norm.` only for that variant; the GroupNorm
            // models keep `norm.gamma`/`norm.beta` untouched.
            //
            // Some pths (4x-alchemy) wrap the state dict under `params`, others
            // (2xPublic) are flat — the reader recurses nested dicts by default,
            // so `^params\.` → "" handles both (no-op on flat files).
            //
            // NOTE: the pth must have contiguous tensors — burn-store's reader
            // ignores strides (docs/upstream-issues.md §4), so a channels-last
            // state dict (e.g. the raw `4x_Alchemy.pth`) loads scrambled.
            // Preprocess with `{k: v.contiguous() for k, v in sd.items()}`.
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
            // Phhofm is flat; TNTwise wraps in `params` (stripped). Stale
            // `eval_conv.*` and `no_norm` are ignored by `load_from`. The 5th
            // CLI arg (num_block slot) is the feature-channel count: 48 for
            // the Phhofm 2× family, 64 for TNTwise ModernSpanimation V1/V1.5.
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
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}

/// One-time ONNX → f16 `.bpk` conversion (maintainer + `download_model`).
///
/// Reads only the `initializer` tensors via the built-in protobuf reader (no
/// ONNX Runtime); the names already match the module state dict apart from the
/// torch `.conv.0` / `.conv.2` quirk, which is remapped here. Weights are
/// decoded to f32 and saved through `HalfPrecisionAdapter` like the `.pth` path.
pub fn convert_onnx_to_bpk(
    arch: &str,
    onnx_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
) -> Result<()> {
    let bytes = std::fs::read(onnx_path)?;
    let tensors = crate::onnx::read_initializers(&bytes).map_err(Error::new)?;
    let mut snapshots = Vec::with_capacity(tensors.len());
    for t in tensors {
        let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let data = onnx_data_to_f32(&t)?;
        let mut s = TensorSnapshot::from_data(
            TensorData::new(data, shape),
            t.name.split('.').map(str::to_string).collect(),
            Vec::new(),
            ParamId::new(),
        );
        s.container_stack = None;
        s.tensor_id = None;
        snapshots.push(s);
    }
    let remapper = KeyRemapper::from_patterns(vec![
        (r"\.conv\.0\.", ".conv."),
        (r"\.conv\.2\.", ".conv2."),
    ])
    .map_err(|e| Error::new(e.to_string()))?;
    let (snapshots, _) = remapper.remap(snapshots);

    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(HalfPrecisionAdapter::new());
    match arch {
        "upcunet2x" => {
            let mut m = UpCunet2x::<BurnBackend>::new(&device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        "upcunet2x-fast" | "fallin-cugan" => {
            let mut m = UpCunet2xFast::<BurnBackend>::new(&device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        "realesrgan" => {
            let mut m = RrdbNet::<BurnBackend>::new(scale as usize, num_block as usize, &device);
            apply_and_save(&mut m, snapshots, &mut save)?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}

fn apply_and_save<B, M>(
    m: &mut M,
    snapshots: Vec<TensorSnapshot>,
    save: &mut BurnpackStore,
) -> Result<()>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let result = m.apply(snapshots, None, None, true);
    if !result.missing.is_empty() {
        return Err(Error::new(format!("missing tensors:\n{result}")));
    }
    m.save_into(save).map_err(|e| Error::new(e.to_string()))?;
    Ok(())
}

fn onnx_data_to_f32(t: &crate::onnx::OnnxTensor) -> Result<Vec<f32>> {
    let n = t.dims.iter().map(|&d| d as usize).product::<usize>();
    let mut out = Vec::with_capacity(n);
    match t.dtype {
        1 => {
            for c in t.data.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        10 => {
            for c in t.data.chunks_exact(2) {
                out.push(f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32());
            }
        }
        11 => {
            for c in t.data.chunks_exact(8) {
                out.push(
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                );
            }
        }
        6 => {
            for c in t.data.chunks_exact(4) {
                out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        7 => {
            for c in t.data.chunks_exact(8) {
                out.push(
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                );
            }
        }
        other => {
            return Err(Error::new(format!(
                "unsupported ONNX dtype {other} for {}",
                t.name
            )))
        }
    }
    if out.len() != n {
        return Err(Error::new(format!("data length mismatch for {}", t.name)));
    }
    Ok(out)
}

