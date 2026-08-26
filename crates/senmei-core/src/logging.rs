//! Transport-agnostic log helpers shared by the GUI (Tauri) and headless
//! (HTTP) adapters: the `LogEntry` shape, a bounded ring buffer for a Logs
//! panel, and a rotating file appender. Adapters only differ in delivery
//! (Tauri event vs HTTP poll) and console routing.

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Ring-buffer capacity for the in-memory Logs-panel history.
pub const BUFFER_CAP: usize = 1000;
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_ROTATIONS: usize = 3;

/// One log line for a Logs panel (GUI + HTTP share the shape).
#[derive(Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct LogEntry {
    pub level: String,
    pub message: String,
    pub timestamp: u64,
}

impl LogEntry {
    pub fn new(level: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level: level.into(),
            message: message.into(),
            timestamp: epoch_ms(),
        }
    }
}

/// Bounded in-memory log history for a Logs panel.
#[derive(Default)]
pub struct LogBuffer {
    inner: Mutex<VecDeque<LogEntry>>,
}

impl LogBuffer {
    pub fn push(&self, entry: LogEntry) {
        let mut g = self.inner.lock().unwrap();
        if g.len() >= BUFFER_CAP {
            g.pop_front();
        }
        g.push_back(entry);
    }

    pub fn entries(&self) -> Vec<LogEntry> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }
}

pub fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `HH:MM:SS.mmm` from epoch ms.
pub fn fmt_ts(ms: u64) -> String {
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

/// Append one line to `<dir>/senmei.log`, rotating once it outgrows the cap.
pub fn append_rotating(dir: &Path, line: &str) {
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = dir.join("senmei.log");
    if fs::metadata(&path).map(|m| m.len()).unwrap_or(0) >= LOG_MAX_BYTES {
        rotate(&path);
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Shift `.log.{i}` → `.log.{i+1}` (current → `.log.1`), dropping the oldest.
fn rotate(path: &Path) {
    for i in (0..LOG_ROTATIONS).rev() {
        let src = if i == 0 {
            path.to_path_buf()
        } else {
            path.with_file_name(format!("senmei.log.{i}"))
        };
        if src.exists() {
            let dst = path.with_file_name(format!("senmei.log.{}", i + 1));
            // Windows `rename` fails when the target exists — clear it first.
            let _ = fs::remove_file(&dst);
            let _ = fs::rename(&src, dst);
        }
    }
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

        rotate(&path);

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
