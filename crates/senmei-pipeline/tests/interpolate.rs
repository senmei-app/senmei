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

    let info = senmei_media::probe(&output).unwrap();
    assert!((info.fps - 20.0).abs() < 1.0, "expected ~20 fps, got {}", info.fps);

    let _ = std::fs::remove_dir_all(&dir);
}
