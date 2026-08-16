#[tauri::command]
pub fn health_check() -> String {
    "ok".to_string()
}
