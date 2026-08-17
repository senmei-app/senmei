//! Smoke test for the preview backend functions.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn preview_backend_functions_work() {
    let dir = std::env::temp_dir().join("senmei-preview-smoke");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input: PathBuf = dir.join("input.mp4");
    let ok = Command::new("ffmpeg")
        .args(["-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x180:rate=24", "-pix_fmt", "yuv420p"])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg input generation failed");

    let info = senmei_media::probe(&input).expect("probe failed");
    assert!(info.width > 0 && info.duration > 0.0);

    let jpeg = senmei_media::extract_frame(&input, 0.5).expect("extract_frame failed");
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "not a JPEG");

    let _ = std::fs::remove_dir_all(&dir);
}
