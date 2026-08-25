//! One-off helper: resolve (download) the runtime libtorch into a data dir.
//! Run: cargo run -p senmei-ml --example resolve_libtorch -- <data_dir>

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/mzach/.local/share/senmei".to_string());
    let hw = senmei_ml::detect();
    eprintln!("variant: {:?}", senmei_ml::pick_variant(&hw));
    match senmei_ml::resolve(std::path::Path::new(&data_dir), &hw) {
        Ok(Some(install)) => {
            println!(
                "resolved: {:?} lib={}",
                install.variant,
                install.lib_dir.display()
            );
        }
        Ok(None) => println!("no CUDA/ROCm device detected"),
        Err(e) => {
            eprintln!("resolve failed: {e}");
            std::process::exit(1);
        }
    }
}
