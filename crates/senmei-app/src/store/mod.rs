//! App-local persistence: settings (JSON in the data dir) and project
//! management (folders + `projects.json` index, tar.xz import/export).

mod projects;
mod settings;

pub use projects::{
    create_project, delete_project, ensure_within_data_dir, export_project, list_projects,
    load_project_settings, open_project, save_project_settings, ProjectEntry, ProjectSettings,
};
pub use settings::{load_settings, save_settings, Settings};

use std::path::PathBuf;

/// App data dir (`$XDG_DATA_HOME/senmei`), same convention as the server core.
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

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::projects::PipelineStep;
    use super::*;
    use std::collections::HashMap;

    fn with_temp_data_dir(name: &str, test: impl FnOnce()) {
        let _guard = super::TEST_ENV_LOCK.lock().unwrap();
        let base =
            std::env::temp_dir().join(format!("senmei-store-test-{}-{name}", std::process::id()));
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
                hotkeys: Some(HashMap::from([("render".into(), "Ctrl+Shift+R".into())])),
                tile_size: Some(512),
                pipeline_depth: Some(2),
                backend: Some(senmei_ml::EngineBackend::Vulkan),
                gpu_index: Some(1),
            };
            save_settings(&settings).unwrap();
            let loaded = load_settings();
            assert_eq!(loaded.language, "de");
            assert_eq!(loaded.theme, "light");
            assert_eq!(loaded.tile_size, Some(512));
            assert_eq!(loaded.pipeline_depth, Some(2));
            assert_eq!(loaded.backend, Some(senmei_ml::EngineBackend::Vulkan));
            assert_eq!(
                loaded.hotkeys.as_ref().and_then(|h| h.get("render")),
                Some(&"Ctrl+Shift+R".to_string())
            );
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
    fn ensure_within_data_dir_refuses_outside() {
        with_temp_data_dir("allowlist", || {
            ensure_within_data_dir(&data_dir().join("projects")).unwrap();
            let outside = std::env::temp_dir().join("senmei-allowlist-outside");
            let err = ensure_within_data_dir(&outside).unwrap_err();
            assert!(err.contains("refusing"), "unexpected error: {err}");
        });
    }

    #[test]
    fn open_project_refuses_tar_slip() {
        with_temp_data_dir("tarslip", || {
            // Hand-craft a raw tar entry with a `..` path: the tar Builder
            // refuses to write these, so this exercises the unpack guard.
            let mut h = [0u8; 512];
            h[..13].copy_from_slice(b"../escape.txt");
            h[100..108].copy_from_slice(b"0000644\0");
            h[108..116].copy_from_slice(b"0000000\0");
            h[116..124].copy_from_slice(b"0000000\0");
            h[124..136].copy_from_slice(b"00000000000\0"); // size 0
            h[136..148].copy_from_slice(b"00000000000\0"); // mtime 0
            h[148..156].copy_from_slice(b"        "); // checksum placeholder
            h[156] = b'0'; // typeflag: regular file
            h[257..263].copy_from_slice(b"ustar\0");
            h[263..265].copy_from_slice(b"00");
            let sum: u32 = h.iter().map(|&b| b as u32).sum();
            h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
            let mut tar_bytes = h.to_vec();
            tar_bytes.extend_from_slice(&[0u8; 1024]); // EOF blocks

            let archive =
                std::env::temp_dir().join(format!("senmei-tarslip-{}.tar.xz", std::process::id()));
            let file = std::fs::File::create(&archive).unwrap();
            let mut xz = liblzma::write::XzEncoder::new(file, 6);
            use std::io::Write;
            xz.write_all(&tar_bytes).unwrap();
            xz.finish().unwrap();

            let err = open_project(&archive.to_string_lossy()).unwrap_err();
            assert!(!err.is_empty(), "expected tar-slip refusal");
            let _ = std::fs::remove_file(&archive);
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
