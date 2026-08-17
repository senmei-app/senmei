use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub language: String,
    pub theme: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: "en".into(),
            theme: "dark".into(),
        }
    }
}

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
pub struct ProjectSettings {
    #[serde(default)]
    pub steps: Vec<PipelineStep>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub output_dir: Option<String>,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            files: Vec::new(),
            output_dir: None,
        }
    }
}

pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local")
                .join("share")
        });
    base.join("senmei")
}

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

fn projects_path() -> PathBuf {
    data_dir().join("projects.json")
}

pub fn load_settings() -> Settings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_settings(settings: &Settings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

fn project_settings_path(project_dir: &PathBuf) -> PathBuf {
    project_dir.join("project.json")
}

pub fn load_project_settings(project_dir: &PathBuf) -> ProjectSettings {
    std::fs::read_to_string(project_settings_path(project_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_project_settings(
    project_dir: &PathBuf,
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
        tar.append_path_with_name(&path, &name).map_err(|e| e.to_string())?;
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
    let safe = if safe.trim().is_empty() { "project" } else { safe.trim() };

    let base = data_dir().join("projects").join(safe);
    let target = unique_dir(&base);
    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let xz = liblzma::read::XzDecoder::new(file);
    let mut ar = tar::Archive::new(xz);
    ar.unpack(&target).map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_temp_data_dir(name: &str, test: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().unwrap();
        let base = std::env::temp_dir().join(format!(
            "senmei-store-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::env::set_var("XDG_DATA_HOME", &base);
        test();
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn settings_roundtrip() {
        with_temp_data_dir("roundtrip", || {
            let settings = Settings {
                language: "de".into(),
                theme: "light".into(),
            };
            save_settings(&settings).unwrap();
            let loaded = load_settings();
            assert_eq!(loaded.language, "de");
            assert_eq!(loaded.theme, "light");
        });
    }

    #[test]
    fn settings_default_when_missing() {
        with_temp_data_dir("defaults", || {
            let loaded = load_settings();
            assert_eq!(loaded.language, "en");
            assert_eq!(loaded.theme, "dark");
        });
    }

    #[test]
    fn project_dir_created_by_create_project() {
        with_temp_data_dir("project", || {
            let path = create_project("Test 1").unwrap();
            assert!(PathBuf::from(&path).is_dir());
        });
    }

    #[test]
    fn delete_project_removes_dir_and_forgets() {
        with_temp_data_dir("delete_project", || {
            let path = create_project("DeleteMe").unwrap();
            assert!(PathBuf::from(&path).is_dir());
            delete_project(&path).unwrap();
            assert!(!PathBuf::from(&path).exists());
            assert!(!list_projects().iter().any(|p| p.path == path));
        });
    }

    #[test]
    fn delete_project_refuses_outside_dir() {
        with_temp_data_dir("delete_outside", || {
            let outside = std::env::temp_dir().join("senmei-outside-dir");
            std::fs::create_dir_all(&outside).unwrap();
            let err = delete_project(&outside.to_string_lossy()).unwrap_err();
            assert!(err.contains("refusing"), "unexpected error: {err}");
            let _ = std::fs::remove_dir_all(&outside);
        });
    }

    #[test]
    fn export_open_project_roundtrip() {
        with_temp_data_dir("export_open", || {
            let dir = PathBuf::from(create_project("RoundTrip").unwrap());
            let mut settings = ProjectSettings::default();
            settings.files = vec!["/videos/a.mp4".into(), "/videos/b.mp4".into()];
            save_project_settings(&dir, &settings).unwrap();

            let archive = std::env::temp_dir().join("senmei-roundtrip.tar.xz");
            let _ = std::fs::remove_file(&archive);
            export_project(&dir.to_string_lossy(), &archive.to_string_lossy()).unwrap();
            assert!(archive.exists() && archive.metadata().unwrap().len() > 0);

            let imported = PathBuf::from(open_project(&archive.to_string_lossy()).unwrap());
            assert_ne!(imported, dir); // imported into a fresh project dir
            let loaded = load_project_settings(&imported);
            assert_eq!(loaded.files, settings.files);
            let _ = std::fs::remove_file(&archive);
        });
    }

    #[test]
    fn project_settings_roundtrip() {
        with_temp_data_dir("project_settings", || {
            let dir = PathBuf::from(create_project("Settings").unwrap());
            let mut settings = ProjectSettings::default();
            settings.steps.push(PipelineStep {
                id: "1".into(),
                step_type: "upscale".into(),
                enabled: true,
                params: Default::default(),
            });
            settings.steps.push(PipelineStep {
                id: "2".into(),
                step_type: "resize".into(),
                enabled: false,
                params: Default::default(),
            });
            save_project_settings(&dir, &settings).unwrap();

            let loaded = load_project_settings(&dir);
            assert_eq!(loaded.steps.len(), 2);
            assert!(loaded.steps[0].enabled);
            assert!(!loaded.steps[1].enabled);
        });
    }
}
