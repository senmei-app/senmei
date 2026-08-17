//! Maintainer tool: convert a torch `.pth` to the app's f16 `.bpk` burnpack.
//!
//! usage: senmei-ml-convert <arch> <model.pth> <out.bpk> [scale] [num_block]
//!   arch: upcunet2x | upcunet2x-fast | realesrgan
//!   scale / num_block only matter for `realesrgan` (RRDBNet).

fn main() -> senmei_ml::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: senmei-ml-convert <arch> <model.pth> <out.bpk> [scale] [num_block]");
        eprintln!("  arch: upcunet2x | upcunet2x-fast | realesrgan");
        std::process::exit(2);
    }
    let scale: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_block: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
    senmei_ml::convert_pth_to_bpk(
        &args[1],
        std::path::Path::new(&args[2]),
        std::path::Path::new(&args[3]),
        scale,
        num_block,
    )?;
    println!("converted {} -> {}", args[1], args[3]);
    Ok(())
}
