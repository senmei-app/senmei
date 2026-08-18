mod commands;
pub mod log_hub;
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
            commands::delete_project,
            commands::export_project,
            commands::open_project,
            commands::load_project_settings,
            commands::save_project_settings,
            commands::get_ffmpeg_status,
            commands::download_ffmpeg,
            commands::list_models,
            commands::download_model,
            commands::probe_video,
            commands::read_frame,
            commands::cancel_render,
            commands::pause_render,
            commands::prune_samples,
            commands::unique_path,
            log_hub::get_logs
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
