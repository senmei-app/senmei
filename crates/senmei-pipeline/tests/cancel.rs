use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
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
