use std::path::PathBuf;
use std::process::Command;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn interpolation_doubles_frame_rate() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-m3-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let input: PathBuf = dir.join("input.mp4");
    let output: PathBuf = dir.join("output.mp4");

    let status = Command::new("ffmpeg")
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
        .unwrap();
    assert!(status.success(), "failed to generate test input");

    let steps: Vec<Box<dyn senmei_pipeline::Step>> = vec![Box::new(senmei_pipeline::Passthrough)];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    pipeline.set_interpolator(senmei_pipeline::Interpolator::new(2));
    let ffmpeg = senmei_media::resolve(&dir);

    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let info = senmei_media::probe(&senmei_media::ffprobe_next_to(&ffmpeg), &output).unwrap();
    assert!(
        (info.fps - 20.0).abs() < 1.0,
        "expected ~20 fps, got {}",
        info.fps
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires Vulkan + models/flownet.bin; needs RUST_MIN_STACK=33554432"]
fn rife_interpolates_real_model_e2e() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }
    let models_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
    let bin = models_dir.join("flownet.bin");
    if !bin.exists() {
        eprintln!("missing flownet.bin, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-rife-e2e");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input: PathBuf = dir.join("input.mp4");
    let output: PathBuf = dir.join("output.mp4");

    let status = Command::new("ffmpeg")
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
        .unwrap();
    assert!(status.success(), "failed to generate test input");

    // Load the real RIFE v4.6 weights (burn-Vulkan fp16) like the app does.
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry.resolve("rife-v4.6", &models_dir).unwrap();
    let mut engine = senmei_ml::engine_for_model(
        &mref,
        senmei_ml::EngineBackend::default(),
        &std::env::temp_dir(),
    )
    .unwrap();
    engine.load(&mref).unwrap();

    let steps: Vec<Box<dyn senmei_pipeline::Step>> = vec![Box::new(senmei_pipeline::Passthrough)];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    pipeline.set_interpolator(senmei_pipeline::Interpolator::with_engine(2, engine));
    let ffmpeg = senmei_media::resolve(&dir);

    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let info = senmei_media::probe(&senmei_media::ffprobe_next_to(&ffmpeg), &output).unwrap();
    assert!(
        (info.fps - 20.0).abs() < 1.0,
        "expected ~20 fps, got {}",
        info.fps
    );

    // Input had 10 frames; factor-2 interpolation emits 10 + 9 = 19 frames.
    let count = Command::new(senmei_media::ffprobe_next_to(&ffmpeg))
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .unwrap();
    let frames: f64 = String::from_utf8_lossy(&count.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0);
    assert_eq!(frames, 19.0, "expected 19 frames, got {frames}");

    let _ = std::fs::remove_dir_all(&dir);
}
