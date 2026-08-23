//! Maintainer tool: convert a torch `.pth` or ONNX model to the app's f16
//! `.bpk` burnpack.
//!
//! usage: senmei-ml-convert <arch> <model> <out.bpk> [scale] [num_block] [layer_norm] [dysample]
//!   arch: upcunet2x | upcunet2x-fast | fallin-cugan | realesrgan | real-plksr
//!         | ifrnet | drunet | dncnn | ffdnet | nafnet | span | safmn
//!   model: a `.pth` state dict or an `.onnx` file (initializers are read via
//!          the built-in parser — no ONNX Runtime)
//!   scale / num_block only matter for `realesrgan` (RRDBNet) and `real-plksr`
//!          (scale: 1 for the decompress models, 4 for 4x-alchemy).
//!   for `span`, the 5th arg is the feature-channel count: 48 for the Phhofm
//!          2× family, 64 for TNTwise ModernSpanimation V1/V1.5.
//!   for `real-plksr`, the 6th arg toggles the channel LayerNorm variant
//!          (`layer_norm=1`, e.g. `real-plksr-2x-public`) and the 7th the
//!          DySample tail (`dysample=0` for the pixel-shuffle tail, e.g.
//!          `4x-nomoswebphoto`).
//!   for `realesrgan`, the 8th arg is the shuffle factor (2 for the
//!          pixel-unshuffled RealESRGAN_x2plus variant, default 1).
//!
//! `real-plksr` pths must have contiguous tensors (burn-store ignores strides,
//! see docs/upstream-issues.md §4 — preprocess channels-last state dicts with
//! `{k: v.contiguous() for k, v in sd.items()}`.

fn main() -> senmei_ml::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: senmei-ml-convert <arch> <model.pth|model.onnx> <out.bpk> [scale] [num_block] [layer_norm] [dysample]"
        );
        eprintln!(
            "  arch: upcunet2x | upcunet2x-fast | fallin-cugan | realesrgan | real-plksr | ifrnet | drunet | dncnn | ffdnet | nafnet | span | safmn"
        );
        std::process::exit(2);
    }
    let scale: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_block: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
    let layer_norm = args
        .get(6)
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let dysample = args
        .get(7)
        .map(|s| !(s == "0" || s.eq_ignore_ascii_case("false")))
        .unwrap_or(true);
    let shuffle: u32 = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(1);
    let input = std::path::Path::new(&args[2]);
    let out = std::path::Path::new(&args[3]);
    if input.extension().and_then(|e| e.to_str()) == Some("onnx") {
        senmei_ml::convert_onnx_to_bpk(&args[1], input, out, scale, num_block, shuffle)?;
    } else {
        senmei_ml::convert_pth_to_bpk(
            &args[1], input, out, scale, num_block, layer_norm, dysample, shuffle,
        )?;
    }
    println!("converted {} -> {}", args[1], args[3]);
    Ok(())
}
