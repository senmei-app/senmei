//! File-backed logging for headless/HTTP runs: `env_logger` to stdout plus a
//! rotating `senmei.log` (Info+) in the app data dir — same scheme as the GUI,
//! so crashes and HTTP errors survive without a visible terminal.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_ROTATIONS: usize = 3;

struct FileSink {
    file: Mutex<fs::File>,
}

fn rotated(path: &Path, n: usize) -> PathBuf {
    path.with_file_name(format!("senmei.log.{n}"))
}

fn rotate(path: &Path) {
    for i in (0..LOG_ROTATIONS).rev() {
        let src = if i == 0 {
            path.to_path_buf()
        } else {
            rotated(path, i)
        };
        if src.exists() {
            let _ = fs::rename(&src, rotated(path, i + 1));
        }
    }
}

/// Install the server logger (idempotent): stdout via env_logger + a rotating
/// file in `data_dir/logs`. Safe to call once.
pub fn init(data_dir: &Path) {
    let logs_dir = data_dir.join("logs");
    let _ = fs::create_dir_all(&logs_dir);
    let path = logs_dir.join("senmei.log");
    if path.metadata().map(|m| m.len()).unwrap_or(0) > LOG_MAX_BYTES {
        rotate(&path);
    }
    let sink = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()
        .map(|f| FileSink { file: Mutex::new(f) });
    let console = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .build();
    let _ = log::set_boxed_logger(Box::new(ServerLogger { console, sink }));
    log::set_max_level(log::LevelFilter::Info);
}

struct ServerLogger {
    console: env_logger::Logger,
    sink: Option<FileSink>,
}

impl log::Log for ServerLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        self.console.log(record);
        if let Some(sink) = &self.sink {
            let line = format!("[{} {}] {}\n", stamp(), record.level(), record.args());
            let mut f = sink.file.lock().unwrap();
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }
    fn flush(&self) {}
}

fn stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}
