//! Forward `log` records to the webview Logs panel and a rotating log file:
//! a bounded in-memory buffer + a Tauri event; the file (Info+) lives in the
//! app data dir so logs survive crashes. Console stays error-only +
//! `wgpu_hal=off`.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

const BUFFER_CAP: usize = 500;
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_ROTATIONS: usize = 3;

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
    log_dir: Option<PathBuf>,
}

fn hub() -> &'static Arc<Mutex<Hub>> {
    static HUB: OnceLock<Arc<Mutex<Hub>>> = OnceLock::new();
    HUB.get_or_init(|| {
        Arc::new(Mutex::new(Hub {
            entries: VecDeque::with_capacity(BUFFER_CAP),
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
                append_log(dir, &line);
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

/// Append one line to `<dir>/senmei.log`, rotating once it outgrows the cap.
fn append_log(dir: &Path, line: &str) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("senmei.log");
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) >= LOG_MAX_BYTES {
        rotate_logs(&path);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Shift `.log.{i-1}` → `.log.{i}`, then the current file → `.log.1`.
fn rotate_logs(path: &Path) {
    for i in (2..=LOG_ROTATIONS).rev() {
        let from = path.with_file_name(format!("senmei.log.{}", i - 1));
        let to = path.with_file_name(format!("senmei.log.{i}"));
        let _ = std::fs::rename(&from, &to);
    }
    let _ = std::fs::rename(path, &path.with_file_name("senmei.log.1"));
}

/// `HH:MM:SS.mmm` from epoch ms.
fn fmt_ts(ms: u64) -> String {
    let secs = ms / 1000;
    let millis = ms % 1000;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        (secs / 3600) % 24,
        (secs / 60) % 60,
        secs % 60,
        millis,
    )
}

/// Buffered entries for the Logs panel when it opens.
#[tauri::command]
#[specta::specta]
pub fn get_logs() -> Vec<LogEntry> {
    hub().lock().unwrap().entries.iter().cloned().collect()
}

/// Empty the buffered log history (Logs panel "Clear").
#[tauri::command]
#[specta::specta]
pub fn clear_logs() {
    hub().lock().unwrap().entries.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_shifts_backups_and_drops_oldest() {
        let dir = std::env::temp_dir().join(format!("senmei-log-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("senmei.log");
        std::fs::write(&path, b"current").unwrap();
        std::fs::write(dir.join("senmei.log.1"), b"one").unwrap();
        std::fs::write(dir.join("senmei.log.2"), b"two").unwrap();

        rotate_logs(&path);

        assert!(!path.exists());
        assert_eq!(std::fs::read(dir.join("senmei.log.1")).unwrap(), b"current");
        assert_eq!(std::fs::read(dir.join("senmei.log.2")).unwrap(), b"one");
        assert_eq!(std::fs::read(dir.join("senmei.log.3")).unwrap(), b"two");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fmt_ts_pads() {
        assert_eq!(fmt_ts(0), "00:00:00.000");
        assert_eq!(fmt_ts(3723_456), "01:02:03.456");
    }
}
