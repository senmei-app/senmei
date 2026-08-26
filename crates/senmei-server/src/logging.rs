//! File-backed logging for headless/HTTP runs: `env_logger` to stdout plus a
//! rotating `senmei.log` (Info+) in the app data dir — same scheme as the GUI,
//! so crashes and HTTP errors survive without a visible terminal.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use senmei_core::logging::{append_rotating, fmt_ts, LogBuffer, LogEntry};

/// In-memory ring buffer feeding the web UI Logs panel over HTTP.
fn buffer() -> &'static LogBuffer {
    static BUF: OnceLock<LogBuffer> = OnceLock::new();
    BUF.get_or_init(LogBuffer::default)
}

/// Buffered entries for the web UI Logs panel when it opens.
pub fn entries() -> Vec<LogEntry> {
    buffer().entries()
}

/// Empty the buffered log history (Logs panel "Clear").
pub fn clear() {
    buffer().clear();
}

/// Install the server logger (idempotent): stdout via env_logger + a rotating
/// file in `data_dir/logs`. Safe to call once.
pub fn init(data_dir: &Path) {
    let logs_dir = data_dir.join("logs");
    let console =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let _ = log::set_boxed_logger(Box::new(ServerLogger { console, logs_dir }));
    log::set_max_level(log::LevelFilter::Info);
}

struct ServerLogger {
    console: env_logger::Logger,
    logs_dir: PathBuf,
}

impl log::Log for ServerLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let entry = LogEntry::new(record.level().to_string(), record.args().to_string());
        buffer().push(entry.clone());
        self.console.log(record);
        let line = format!(
            "[{} {}] {}",
            fmt_ts(entry.timestamp),
            record.level(),
            record.args()
        );
        append_rotating(&self.logs_dir, &line);
    }
    fn flush(&self) {}
}
