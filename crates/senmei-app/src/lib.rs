mod commands;
mod store;

use specta_typescript::Typescript;
use tauri_specta::{collect_commands, Builder, ErrorHandlingMode};

const BINDINGS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../packages/bridge/src/bindings.ts"
);

fn specta_builder() -> Builder<tauri::Wry> {
    Builder::<tauri::Wry>::new()
        .dangerously_cast_bigints_to_number()
        .error_handling(ErrorHandlingMode::Throw)
        .commands(collect_commands![
            commands::health_check,
            commands::render,
            commands::import_folder,
            commands::get_settings,
            commands::save_settings,
            commands::list_projects,
            commands::create_project,
            commands::remember_project,
            commands::get_ffmpeg_status,
            commands::download_ffmpeg,
            commands::list_models,
            commands::get_libtorch_status,
            commands::download_libtorch
        ])
}

pub fn run() {
    let builder = specta_builder();

    #[cfg(debug_assertions)]
    builder
        .export(Typescript::default(), BINDINGS_PATH)
        .expect("failed to export typescript bindings");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    #[test]
    fn export_ts_bindings() {
        super::specta_builder()
            .export(specta_typescript::Typescript::default(), super::BINDINGS_PATH)
            .expect("failed to export typescript bindings");
    }
}
