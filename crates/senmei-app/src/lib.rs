mod commands;

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![commands::health_check, commands::render])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
