//! ONNX → f16 `.bpk` conversion.

use super::ToF16;
use crate::arch::{RrdbNet, UpCunet2x, UpCunet2xFast};
use crate::BurnBackend;
use crate::{Error, Result};
use burn::module::ParamId;
use burn::tensor::backend::Backend;
use burn::tensor::{f16, TensorData};
use burn_store::{BurnpackStore, KeyRemapper, ModuleSnapshot, TensorSnapshot};
use burn_wgpu::WgpuDevice;
use std::path::Path;

/// Built-in protobuf reader (no ONNX Runtime). Remaps torch `.conv.0`/`.conv.2`.
pub fn convert_onnx_to_bpk(
    arch: &str,
    onnx_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
    shuffle: u32,
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
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(ToF16);
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
            let mut m = RrdbNet::<BurnBackend>::new(
                scale as usize,
                num_block as usize,
                shuffle as usize,
                &device,
            );
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
