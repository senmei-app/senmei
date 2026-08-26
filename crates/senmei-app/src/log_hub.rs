//! Forward `log` records to the webview Logs panel and a rotating log file:
//! a bounded in-memory buffer + a Tauri event; the file (Info+) lives in the
//! app data dir so logs survive crashes. Console stays error-only +
//! `wgpu_hal=off`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use senmei_core::logging::{append_rotating, fmt_ts, LogBuffer, LogEntry};
use tauri::{AppHandle, Emitter};

struct Hub {
    entries: LogBuffer,
    app: Option<AppHandle>,
    log_dir: Option<PathBuf>,
}

fn hub() -> &'static Arc<Mutex<Hub>> {
    static HUB: OnceLock<Arc<Mutex<Hub>>> = OnceLock::new();
    HUB.get_or_init(|| {
        Arc::new(Mutex::new(Hub {
            entries: LogBuffer::default(),
            app: None,
            log_dir: None,
        }))
    })
}

/// Install the log hub as the global logger (idempotent; safe to call once).
pub fn init() {
    let console = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error,wgpu_hal=off"),
    )
    .build();
    let _ = log::set_boxed_logger(Box::new(HubLogger { console }));
    log::set_max_level(log::LevelFilter::Info);
}

struct HubLogger {
    console: env_logger::Logger,
}

impl log::Log for HubLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        let entry = LogEntry::new(record.level().to_string(), record.args().to_string());
        {
            let guard = hub().lock().unwrap();
            guard.entries.push(entry.clone());
            if let Some(dir) = guard.log_dir.as_deref() {
                let line = format!(
                    "[{} {} {}] {} ({}:{})",
                    fmt_ts(entry.timestamp),
                    entry.level,
                    record.module_path().unwrap_or("-"),
                    entry.message,
                    record.file().unwrap_or("-"),
                    record.line().unwrap_or(0),
                );
                append_rotating(dir, &line);
            }
            if let Some(app) = guard.app.clone() {
                let _ = app.emit("log", &entry);
            }
        }
        if self.console.enabled(record.metadata()) {
            self.console.log(record);
        }
    }

    fn flush(&self) {
        self.console.flush();
    }
}

/// Attach the app handle once the webview exists so records stream to it, and
/// enable the rotating log file in the app data dir.
pub fn attach(app: &AppHandle) {
    let mut guard = hub().lock().unwrap();
    guard.app = Some(app.clone());
    guard.log_dir = Some(crate::store::data_dir().join("logs"));
}

/// Buffered entries for the Logs panel when it opens.
#[tauri::command]
#[specta::specta]
pub fn get_logs() -> Vec<LogEntry> {
    hub().lock().unwrap().entries.entries()
}

/// Empty the buffered log history (Logs panel "Clear").
#[tauri::command]
#[specta::specta]
pub fn clear_logs() {
    hub().lock().unwrap().entries.clear();
}
