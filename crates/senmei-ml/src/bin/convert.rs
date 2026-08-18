//! Maintainer tool: convert a torch `.pth` or ONNX model to the app's f16
//! `.bpk` burnpack.
//!
//! usage: senmei-ml-convert <arch> <model> <out.bpk> [scale] [num_block]
//!   arch: upcunet2x | upcunet2x-fast | fallin-cugan | realesrgan | real-plksr
//!         | ifrnet | drunet
//!   model: a `.pth` state dict or an `.onnx` file (initializers are read via
//!          the built-in parser — no ONNX Runtime)
//!   scale / num_block only matter for `realesrgan` (RRDBNet) and `real-plksr`
//!          (scale: 1 for the decompress models, 4 for 4x-alchemy).
//!
//! `real-plksr` pths must have contiguous tensors (burn-store ignores strides,
//! see docs/burn-bugs.md Bug 5) — preprocess channels-last state dicts with
//! `{k: v.contiguous() for k, v in sd.items()}`.

fn main() -> senmei_ml::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: senmei-ml-convert <arch> <model.pth|model.onnx> <out.bpk> [scale] [num_block]"
        );
        eprintln!(
            "  arch: upcunet2x | upcunet2x-fast | fallin-cugan | realesrgan | real-plksr | ifrnet"
        );
        std::process::exit(2);
    }
    let scale: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_block: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
    let input = std::path::Path::new(&args[2]);
    let out = std::path::Path::new(&args[3]);
    if input.extension().and_then(|e| e.to_str()) == Some("onnx") {
        senmei_ml::convert_onnx_to_bpk(&args[1], input, out, scale, num_block)?;
    } else {
        senmei_ml::convert_pth_to_bpk(&args[1], input, out, scale, num_block)?;
    }
    println!("converted {} -> {}", args[1], args[3]);
    Ok(())
}
