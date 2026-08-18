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

    assert!(
        frames.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "expected at least one frame"
    );
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

#[test]
fn passthrough_pause_resume() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-pause-test");
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
            "testsrc=duration=8:size=320x240:rate=24",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success(), "failed to generate test input");

    let steps: Vec<Box<dyn senmei_pipeline::Step>> = vec![Box::new(senmei_pipeline::Passthrough)];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);
    let pause = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    pipeline.set_pause(pause.clone());
    let ffmpeg = senmei_media::resolve(&dir);

    let frames = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counter = frames.clone();
    let handle = std::thread::spawn(move || {
        pipeline
            .run(&ffmpeg, &input, &output, move |p| {
                counter.store(p.frames_processed, std::sync::atomic::Ordering::Relaxed);
            })
            .expect("render failed");
    });

    // Let the pipeline reach the frame loop, then pause and verify no progress.
    std::thread::sleep(std::time::Duration::from_millis(200));
    pause.store(true, std::sync::atomic::Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(250));
    let during = frames.load(std::sync::atomic::Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(250));
    let during2 = frames.load(std::sync::atomic::Ordering::Relaxed);
    pause.store(false, std::sync::atomic::Ordering::Relaxed);

    handle.join().unwrap();
    let total = frames.load(std::sync::atomic::Ordering::Relaxed);

    assert!(total > 0, "expected frames to render");
    assert!(during > 0, "expected some frames before pausing");
    assert_eq!(during, during2, "frames must not advance while paused");
    assert!(total >= during, "frames must advance after resume");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn render_only_time_range() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-range-test");
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
    pipeline.set_range(200, Some(700)); // 500 ms of a 1 s / 10 fps clip = 5 frames
    let ffmpeg = senmei_media::resolve(&dir);
    pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap();

    let count = Command::new("ffprobe")
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
    assert_eq!(
        frames, 5.0,
        "expected 5 frames from the 200..700 ms range, got {frames}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
