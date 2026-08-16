mod commands;
mod store;

use tauri_specta::{collect_commands, Builder, ErrorHandlingMode};

pub fn specta_builder() -> Builder<tauri::Wry> {
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
            commands::download_libtorch,
            commands::download_model
        ])
}

#[cfg(test)]
mod tests {
    use super::specta_builder;
    use specta_typescript::Typescript;

    const BINDINGS_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../packages/bridge/src/bindings.ts"
    );

    #[test]
    fn export_ts_bindings() {
        specta_builder()
            .export(Typescript::default(), BINDINGS_PATH)
            .expect("failed to export typescript bindings");
    }
}
