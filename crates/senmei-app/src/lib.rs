mod commands;
mod store;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::health_check,
            commands::render,
            commands::import_folder,
            commands::get_settings,
            commands::save_settings,
            commands::list_projects,
            commands::create_project,
            commands::remember_project,
            commands::get_ffmpeg_status,
            commands::download_ffmpeg
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
