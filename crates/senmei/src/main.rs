fn main() {
    senmei_app::log_hub::init();
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
        .setup(|app| {
            senmei_app::log_hub::attach(app.handle());
            Ok(())
        })
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
