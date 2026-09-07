//! Safetensors → f16 `.bpk` conversion.

use super::ToF16;
use crate::arch::{DisNet, ParagonSrNet};
use crate::BurnBackend;
use crate::{Error, Result, ResultExt};
use burn_store::{BurnpackStore, KeyRemapper, ModuleSnapshot, SafetensorsStore};
use burn_wgpu::WgpuDevice;
use std::path::Path;

/// One-time safetensors → f16 `.bpk` conversion (maintainer + download_model).
/// Phhofm ships fused release weights as safetensors; the keys already match
/// the module state dict apart from the torch `upsampler.0` Sequential index,
/// remapped here. DIS scale-2 weights need the inverse remap (no upsampler
/// index). Saved through [`ToF16`] like the `.pth` path.
pub fn convert_safetensors_to_bpk(
    arch: &str,
    st_path: &Path,
    bpk_path: &Path,
    scale: u32,
    num_block: u32,
) -> Result<()> {
    let device = WgpuDevice::DiscreteGpu(0);
    let mut save = BurnpackStore::from_file(bpk_path).with_to_adapter(ToF16);
    match arch {
        "paragonsr" => {
            let remapper = KeyRemapper::from_patterns(vec![(r"^upsampler\.0\.", "upsampler.")])
                .map_err_str()?;
            let mut store = SafetensorsStore::from_file(st_path).remap(remapper);
            let mut m = ParagonSrNet::<BurnBackend>::new(scale as usize, 24, 3, 2, 1.5, &device);
            m.load_from(&mut store)
                .map_err_str()?;
            m.save_into(&mut save)
                .map_err_str()?;
        }
        "dis" => {
            let remapper = KeyRemapper::from_patterns(super::pth::dis_remap_patterns())
                .map_err_str()?;
            let mut store = SafetensorsStore::from_file(st_path).remap(remapper);
            let mut m = DisNet::<BurnBackend>::new(32, num_block as usize, scale as usize, &device);
            m.load_from(&mut store)
                .map_err_str()?;
            m.save_into(&mut save)
                .map_err_str()?;
        }
        other => return Err(Error::new(format!("unsupported arch: {other}"))),
    }
    Ok(())
}
