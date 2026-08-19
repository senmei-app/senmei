// Release builds run as GUI app (no console window); debug keeps the console for logs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;

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
            // Packaged app: make the bundled model catalog reachable in the
            // writable data dir (dev falls back to the repo checkout).
            let resource_dir = app.path().resource_dir().ok();
            if let Err(e) = senmei_app::models::ensure_catalog(resource_dir.as_deref()) {
                log::warn!("model catalog not materialized: {e}");
            }
            Ok(())
        })
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
