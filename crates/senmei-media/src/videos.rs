//! Directory scan for video files — one source of truth for what counts as a
//! video, shared by the GUI import and the headless scan endpoint.

use std::path::{Path, PathBuf};

/// Extensions treated as video files (lowercase, without dot).
pub const VIDEO_EXTS: [&str; 10] = [
    "mp4", "mkv", "mov", "webm", "avi", "m4v", "ts", "m2ts", "flv", "wmv",
];

/// Collect video files under `dir`; `recursive` also walks subfolders.
pub fn find_videos(dir: &Path, recursive: bool) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect(dir, recursive, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if recursive {
                collect(&path, recursive, out)?;
            }
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if VIDEO_EXTS.contains(&ext.to_ascii_lowercase().as_str()) {
                out.push(path);
            }
        }
    }
    Ok(())
}
