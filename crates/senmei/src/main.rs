fn main() {
    // Skip wgpu_hal: its Vulkan loader prints benign 32-bit ICD/layer scan
    // errors (e.g. leftover NVIDIA/obs JSONs) on every startup.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error,wgpu_hal=off"),
    )
    .init();
    log::info!("Senmei starting");

    let builder = senmei_app::specta_builder();

    #[cfg(debug_assertions)]
    builder
        .export(
            specta_typescript::Typescript::default(),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../packages/bridge/src/bindings.ts"
            ),
        )
        .expect("failed to export typescript bindings");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
