//! Tauri command tests.

use super::*;
use crate::models::engine_for_model;

#[test]
fn preview_commands_produce_raw_frame_and_info() {
    let dir = std::env::temp_dir().join("senmei-cmd-smoke");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=10",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg input generation failed");

    let info =
        senmei_core::core::probe_video(&input.to_string_lossy()).expect("probe_video failed");
    assert_eq!((info.width, info.height), (160, 120));
    assert!(info.duration > 0.0);

    let frame = read_frame_inner(&input.to_string_lossy(), 500.0).expect("read_frame failed");
    assert_eq!(
        (frame.width, frame.height),
        (160, 120),
        "below the 1280 budget"
    );
    assert_eq!(frame.data.len(), 160 * 120 * 3, "raw RGB24 frame bytes");
    let _ = std::fs::remove_dir_all(&dir);
}

/// End-to-end proof of the app's render path: models_dir resolution +
/// BurnEngine load + real 1080p→2160p upscale + ffmpeg encode.
#[test]
#[ignore = "requires burn Vulkan engine + ffmpeg; ~15s render"]
fn app_render_upscales_real_model() {
    let dir = std::env::temp_dir().join("senmei-render-smoke");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let output = dir.join("output.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=1920x1080:rate=24",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg input generation failed");

    let engine = engine_for_model("real-cugan-x2").expect("engine_for_model");
    let ffmpeg = senmei_media::resolve(&store::data_dir());
    let mut steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Passthrough)];
    steps.push(Box::new(senmei_pipeline::Upscale::new(2, Some(engine))));
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    // Custom ffmpeg args must override the default x264 encoder.
    pipeline.set_encoder_args(vec![
        "-c:v".into(),
        "libx265".into(),
        "-crf".into(),
        "18".into(),
        "-preset".into(),
        "ultrafast".into(),
        "-pix_fmt".into(),
        "yuv420p10le".into(),
    ]);
    pipeline
        .run(&ffmpeg, &input, &output, |_| {})
        .expect("render failed");

    let info = senmei_core::core::probe_video(&output.to_string_lossy()).expect("probe output");
    assert_eq!((info.width, info.height), (3840, 2160));
    assert!(output.exists());
    let ffprobe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&ffprobe.stdout);
    assert!(
        stdout.contains("hevc") && stdout.contains("yuv420p10le"),
        "custom args not applied, got {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unique_path_numbers_collisions() {
    let dir = std::env::temp_dir().join("senmei-unique-smoke");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("out.mkv");
    let b = dir.join("out_2.mkv");
    std::fs::write(&a, b"x").unwrap();
    std::fs::write(&b, b"x").unwrap();
    let free = unique_path(a.to_string_lossy().into_owned()).unwrap();
    assert_eq!(free, dir.join("out_3.mkv").to_string_lossy().into_owned());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn prune_samples_keeps_newest_by_mtime() {
    let _guard = crate::store::TEST_ENV_LOCK.lock().unwrap();
    let base = std::env::temp_dir().join(format!("senmei-prune-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::env::set_var("XDG_DATA_HOME", &base);
    let dir = store::data_dir().join("samples");
    std::fs::create_dir_all(&dir).unwrap();

    // The newest file's name sorts first; a lexical prune would wrongly
    // delete it, a mtime prune must keep it.
    let oldest = dir.join("b.mkv");
    let newest = dir.join("a.mkv");
    std::fs::write(&oldest, b"x").unwrap();
    std::fs::write(&newest, b"x").unwrap();
    let t0 = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
    let t1 = t0 + std::time::Duration::from_secs(10);
    let set = |p: &std::path::Path, t: std::time::SystemTime| {
        std::fs::File::options()
            .write(true)
            .open(p)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(t))
            .unwrap();
    };
    set(&oldest, t0);
    set(&newest, t1);

    prune_samples(dir.to_string_lossy().into_owned(), 1).unwrap();
    assert!(newest.exists(), "newest sample was pruned");
    assert!(!oldest.exists(), "oldest sample kept");
    let _ = std::fs::remove_dir_all(&base);
}
