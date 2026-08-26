use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::data_dir;

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ProjectEntry {
    pub name: String,
    pub path: String,
}

/// Typed params per step type. Only the fields relevant to a step's type are set.
#[derive(Debug, Clone, Default, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct StepParams {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub scale: Option<u32>,
    #[serde(default)]
    pub fps_multiplier: Option<u32>,
    /// Denoise box-blur radius (denoise step).
    #[serde(default)]
    pub radius: Option<u32>,
    /// Deblur unsharp-mask amount (deblur step).
    #[serde(default)]
    pub amount: Option<f32>,
    /// Dedup mean-pixel-diff threshold in [0,1] (deduplication step).
    #[serde(default)]
    pub threshold: Option<f32>,
    /// Free-form FFmpeg `-vf` filter graph applied per frame (filter step;
    /// frame-preserving 1:1 only).
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub factor: Option<String>,
    /// Label for an output step (e.g. "Final", "Intermediate").
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub video_codec: Option<String>,
    #[serde(default)]
    pub audio_codec: Option<String>,
    #[serde(default)]
    pub subtitle_mode: Option<String>,
    /// Raw extra ffmpeg arguments for the output encode (e.g. `-c:v libx265 -crf 18`).
    /// Takes precedence per-flag over the structured fields below.
    #[serde(default)]
    pub ffmpeg_args: Option<String>,
    #[serde(default)]
    pub crf: Option<u32>,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub pix_fmt: Option<String>,
    #[serde(default)]
    pub tune: Option<String>,
    /// Encoder backend preference for the output encode: "auto" | "hw" | "sw".
    #[serde(default)]
    pub encoder_backend: Option<String>,
    /// VA-API encode device: "auto" (discrete GPU) | "igpu" (offload encode).
    #[serde(default)]
    pub encode_device: Option<String>,
    /// Encoder quality profile (sets crf + preset as a bundle).
    #[serde(default)]
    pub quality: Option<String>,
    /// Output color metadata tags (primaries / transfer / matrix, e.g. bt2020).
    #[serde(default)]
    pub color_primaries: Option<String>,
    #[serde(default)]
    pub color_transfer: Option<String>,
    #[serde(default)]
    pub color_matrix: Option<String>,
    /// HDR→SDR tonemapping for the decode stage: "auto" | "always" | "off".
    #[serde(default)]
    pub tonemap: Option<String>,
    /// Output container/extension (e.g. "mp4", "mkv", "webm").
    #[serde(default)]
    pub container: Option<String>,
    /// Output folder mode: "input" | "global" | "custom".
    #[serde(default)]
    pub output_mode: Option<String>,
    /// Custom output folder (when output_mode == "custom").
    #[serde(default)]
    pub output_folder: Option<String>,
}

/// One module in the processing stack. Ordered top→bottom = execution order.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct PipelineStep {
    pub id: String,
    pub step_type: String,
    pub enabled: bool,
    #[serde(default)]
    pub params: StepParams,
}

/// Per-project Inspector settings persisted in `<project>/project.json`.
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub struct ProjectSettings {
    #[serde(default)]
    pub steps: Vec<PipelineStep>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub output_dir: Option<String>,
}


fn projects_path() -> PathBuf {
    data_dir().join("projects.json")
}

fn project_settings_path(project_dir: &Path) -> PathBuf {
    project_dir.join("project.json")
}

pub fn load_project_settings(project_dir: &Path) -> ProjectSettings {
    std::fs::read_to_string(project_settings_path(project_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_project_settings(
    project_dir: &Path,
    settings: &ProjectSettings,
) -> Result<(), String> {
    let path = project_settings_path(project_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

pub fn list_projects() -> Vec<ProjectEntry> {
    let mut projects: Vec<ProjectEntry> = std::fs::read_to_string(projects_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let dir = data_dir().join("projects");
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let path = entry.path().to_string_lossy().into_owned();
                if !projects.iter().any(|p| p.path == path) {
                    projects.push(ProjectEntry { name, path });
                }
            }
        }
    }

    projects.sort_by(|a, b| a.name.cmp(&b.name));
    projects
}

pub fn remember_project(path: &str) -> Result<(), String> {
    let mut projects: Vec<ProjectEntry> = std::fs::read_to_string(projects_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    if !projects.iter().any(|p| p.path == path) {
        let name = PathBuf::from(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        projects.push(ProjectEntry {
            name,
            path: path.to_string(),
        });

        let path_file = projects_path();
        if let Some(parent) = path_file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(&projects).map_err(|e| e.to_string())?;
        std::fs::write(path_file, json).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Allowlist guard for IPC file ops: refuse paths outside the app data dir
/// (same pattern as `delete_project`). Canonicalizes so relative paths or
/// symlinks cannot escape the managed dir.
pub fn ensure_within_data_dir(path: &Path) -> Result<(), String> {
    // Canonicalize the data dir too: on macOS /var -> /private/var, so a
    // canonicalized child would not start_with the raw base.
    let base = data_dir().canonicalize().unwrap_or_else(|_| data_dir());
    let resolved = match path.parent() {
        Some(parent) => match parent.canonicalize() {
            Ok(parent_c) => parent_c.join(path.file_name().unwrap_or_default()),
            Err(_) => path.to_path_buf(),
        },
        None => path.to_path_buf(),
    };
    if resolved.starts_with(&base) {
        Ok(())
    } else {
        Err(format!(
            "refusing to operate outside the app data dir: {}",
            path.display()
        ))
    }
}

pub fn create_project(name: &str) -> Result<String, String> {
    let safe: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim();
    if safe.is_empty() {
        return Err("project name is empty".into());
    }

    let path = data_dir().join("projects").join(safe);
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Export a project folder as a `.tar.xz` (project.json + any other files in
/// the project dir), using the same tar + liblzma path as the FFmpeg download.
pub fn export_project(src: &str, dest: &str) -> Result<(), String> {
    let src_dir = PathBuf::from(src);
    if !src_dir.is_dir() {
        return Err(format!("not a project folder: {src}"));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&src_dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    let file = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    let xz = liblzma::write::XzEncoder::new(file, 6);
    let mut tar = tar::Builder::new(xz);
    for path in files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        tar.append_path_with_name(&path, &name)
            .map_err(|e| e.to_string())?;
    }
    let xz = tar.into_inner().map_err(|e| e.to_string())?;
    xz.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Import a project `.tar.xz` into the app's project storage and return the
/// new project dir. The project name comes from the archive filename; a
/// colliding name gets a `_2`, `_3`, … suffix.
pub fn open_project(archive: &str) -> Result<String, String> {
    let stem = PathBuf::from(archive)
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "project".into());
    let stem = stem.strip_suffix(".tar").unwrap_or(&stem).to_string();
    let safe: String = stem
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.trim().is_empty() {
        "project"
    } else {
        safe.trim()
    };

    let base = data_dir().join("projects").join(safe);
    let target = unique_dir(&base);
    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let xz = liblzma::read::XzDecoder::new(file);
    let mut ar = tar::Archive::new(xz);
    // unpack_in refuses absolute paths / symlink escapes and *skips* `..`
    // entries (returns Ok(false)) — treat a skip as a refusal of the archive.
    for entry in ar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.unpack_in(&target).map_err(|e| e.to_string())? {
            return Err("archive contains a path outside the project dir".into());
        }
    }
    remember_project(&target.to_string_lossy())?;
    Ok(target.to_string_lossy().into_owned())
}

fn unique_dir(base: &Path) -> PathBuf {
    if !base.exists() {
        return base.to_path_buf();
    }
    for n in 2.. {
        let candidate = PathBuf::from(format!("{} {n}", base.to_string_lossy()));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

/// Delete a project: forget it in `projects.json` and remove its directory.
/// Refuses paths outside the app's `projects` dir.
pub fn delete_project(path: &str) -> Result<(), String> {
    let dir = data_dir().join("projects");
    let target = PathBuf::from(path);
    if !target.starts_with(&dir) {
        return Err("refusing to delete outside projects dir".into());
    }

    let mut projects: Vec<ProjectEntry> = std::fs::read_to_string(projects_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    projects.retain(|p| p.path != path);
    if let Some(parent) = projects_path().parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(&projects).map_err(|e| e.to_string())?;
    std::fs::write(projects_path(), json).map_err(|e| e.to_string())?;

    if target.is_dir() {
        std::fs::remove_dir_all(&target).map_err(|e| e.to_string())?;
    }
    Ok(())
}
