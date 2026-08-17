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
fn passthrough_decodes_and_encodes() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-m1-test");
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
    let ffmpeg = senmei_media::resolve(&dir);

    // The progress callback runs on the encode thread, so use a shared counter.
    let frames = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counter = frames.clone();
    pipeline
        .run(&ffmpeg, &input, &output, move |p| {
            counter.store(p.frames_processed, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();

    assert!(frames.load(std::sync::atomic::Ordering::Relaxed) > 0, "expected at least one frame");
    assert!(output.exists());
    assert!(output.metadata().unwrap().len() > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn passthrough_copies_audio() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-audio-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let input: PathBuf = dir.join("input.mp4");
    let output: PathBuf = dir.join("output.mkv");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success(), "failed to generate test input");

    let steps: Vec<Box<dyn senmei_pipeline::Step>> = vec![Box::new(senmei_pipeline::Passthrough)];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    pipeline.set_encoder_args(vec!["-c:a".into(), "copy".into()]);
    let ffmpeg = senmei_media::resolve(&dir);
    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&probe.stdout);
    assert_eq!(stdout.trim(), "aac", "audio not copied, got {stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}
