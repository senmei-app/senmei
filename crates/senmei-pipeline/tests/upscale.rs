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
fn upscale_and_resize_produce_expected_dims() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-m2-test");
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

    let steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Upscale::new(2, None))];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    let ffmpeg = senmei_media::resolve(&dir);

    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let info = senmei_media::probe(&senmei_media::ffprobe_next_to(&ffmpeg), &output).unwrap();
    assert_eq!((info.width, info.height), (320, 240));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires Vulkan + models/up2x-no-denoise.pth.f16.bpk (via senmei-ml-convert)"]
fn burn_engine_upscales_real_model() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }
    let models_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
    let bpk = models_dir.join("up2x-no-denoise.pth.f16.bpk");
    if !bpk.exists() {
        eprintln!("missing .bpk, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-burn-e2e");
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
    assert!(status.success());

    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry.resolve("real-cugan-x2", &models_dir).unwrap();
    let mut engine = senmei_ml::engine_for_model(&mref, senmei_ml::EngineBackend::default(), &std::env::temp_dir()).unwrap();
    engine.load(&mref).unwrap();
    let steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Upscale::new(2, Some(engine)))];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    let ffmpeg = senmei_media::resolve(&dir);

    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let info = senmei_media::probe(&senmei_media::ffprobe_next_to(&ffmpeg), &output).unwrap();
    assert_eq!((info.width, info.height), (320, 240));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resize_factor_produces_expected_dims() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-m2-resize-test");
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

    let steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Resize::new(0.5))];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    let ffmpeg = senmei_media::resolve(&dir);

    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let info = senmei_media::probe(&senmei_media::ffprobe_next_to(&ffmpeg), &output).unwrap();
    assert_eq!((info.width, info.height), (80, 60));

    let _ = std::fs::remove_dir_all(&dir);
}
