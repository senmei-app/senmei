use std::path::PathBuf;

fn main() {
    // Tauri mangles `..` in resource paths to `_up_` dirs; copy the model
    // catalog to a `..`-free path so the bundle stays tidy.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest.join("../../models/metadata.json");
    let dst = manifest.join("resources/metadata.json");
    if src.is_file() {
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::copy(&src, &dst).unwrap();
    }
    tauri_build::build()
}
