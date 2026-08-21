//! One-click diagnose bundle: app logs + system info as a `.tar.xz`.

use serde_json::json;

use crate::store;

const LOG_ROTATIONS: [&str; 4] = ["senmei.log", "senmei.log.1", "senmei.log.2", "senmei.log.3"];

/// Package the rotating logs + a `diagnostics.json` summary into `dest`
/// (`.tar.xz`), reusing the project-export tar/liblzma path.
pub fn export(dest: &str) -> Result<(), String> {
    let staging = store::data_dir().join("diagnostics-staging");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

    let mut logs = vec![];
    let logs_dir = store::data_dir().join("logs");
    for name in LOG_ROTATIONS {
        let src = logs_dir.join(name);
        if src.is_file() {
            std::fs::copy(&src, staging.join(name)).map_err(|e| e.to_string())?;
            logs.push(name);
        }
    }

    let info = json!({
        "app": "senmei",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
        "settings": store::load_settings(),
        "backend": senmei_ml::backend_info(),
        "ffmpeg": senmei_media::probe_ffmpeg(&senmei_media::resolve(&store::data_dir())),
        "log_files": logs,
    });
    let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?;
    std::fs::write(staging.join("diagnostics.json"), json).map_err(|e| e.to_string())?;

    let result = store::export_project(&staging.to_string_lossy(), dest);
    let _ = std::fs::remove_dir_all(&staging);
    result
}
