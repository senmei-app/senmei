mod commands;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::health_check,
            commands::render,
            commands::import_folder,
            commands::create_project,
            commands::list_projects
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
