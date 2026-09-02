use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A cancel set before the run must abort the pipeline quickly: the encoder is
/// killed instead of being left to mux the whole file, so `run` returns the
/// cancel error promptly and no ffmpeg child is left finalizing.
#[test]
fn cancel_aborts_pipeline_without_hanging() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not found, skipping");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-cancel-test");
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

    let cancel = Arc::new(AtomicBool::new(true));
    let mut pipeline = senmei_pipeline::Pipeline::new(Vec::new());
    pipeline.set_cancel(cancel);
    let ffmpeg = senmei_media::resolve(&dir);

    let start = Instant::now();
    let err = pipeline.run(&ffmpeg, &input, &output, |_| {}).unwrap_err();
    assert_eq!(err.to_string(), "cancelled");
    // The abort path must not wait for a full mux finalize (which held the
    // pipeline + GPU engine for seconds+ on ranged renders before).
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "cancel took too long: {:?}",
        start.elapsed()
    );
}

/// A pause set before the run must stall the pipeline (no frames advance) and
/// resume to a normal finish once it is cleared.
#[test]
fn pause_stalls_then_resumes() {
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
            "testsrc=duration=1:size=160x120:rate=10",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success(), "failed to generate test input");

    let pause = Arc::new(AtomicBool::new(true));
    let mut pipeline = senmei_pipeline::Pipeline::new(Vec::new());
    pipeline.set_pause(pause.clone());
    let ffmpeg = senmei_media::resolve(&dir);

    // The pause flag is polled between frames on the run thread; clear it from
    // a helper thread so the run can finish.
    let unpause = pause.clone();
    let clearer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        unpause.store(false, Ordering::Relaxed);
    });

    let frames = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counter = frames.clone();
    pipeline
        .run(&ffmpeg, &input, &output, move |p| {
            counter.store(p.frames_processed, Ordering::Relaxed);
        })
        .unwrap();
    clearer.join().unwrap();

    assert!(
        frames.load(Ordering::Relaxed) > 0,
        "expected frames after resume"
    );
    assert!(output.exists());
    let _ = std::fs::remove_dir_all(&dir);
}
