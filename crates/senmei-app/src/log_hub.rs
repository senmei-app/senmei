//! Forward `log` records to the webview Logs panel: a bounded buffer plus a
//! Tauri event. Console output keeps its env_logger behavior (error-only +
//! `wgpu_hal=off` by default); the panel captures Info+ regardless.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

const BUFFER_CAP: usize = 500;

#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct LogEntry {
    pub level: String,
    pub message: String,
    pub timestamp: u64,
}

struct Hub {
    entries: VecDeque<LogEntry>,
    app: Option<AppHandle>,
}

fn hub() -> &'static Arc<Mutex<Hub>> {
    static HUB: OnceLock<Arc<Mutex<Hub>>> = OnceLock::new();
    HUB.get_or_init(|| {
        Arc::new(Mutex::new(Hub {
            entries: VecDeque::with_capacity(BUFFER_CAP),
            app: None,
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
        let entry = LogEntry {
            level: record.level().to_string(),
            message: record.args().to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        {
            let mut guard = hub().lock().unwrap();
            if guard.entries.len() >= BUFFER_CAP {
                guard.entries.pop_front();
            }
            guard.entries.push_back(entry.clone());
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

/// Attach the app handle once the webview exists so records stream to it.
pub fn attach(app: &AppHandle) {
    hub().lock().unwrap().app = Some(app.clone());
}

/// Buffered entries for the Logs panel when it opens.
#[tauri::command]
#[specta::specta]
pub fn get_logs() -> Vec<LogEntry> {
    hub().lock().unwrap().entries.iter().cloned().collect()
}
