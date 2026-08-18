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

    let rgb = vec![128u8; (info.width * info.height * 3) as usize];
    let png = senmei_media::encode_png(info.width, info.height, &rgb).expect("encode_png failed");
    assert!(png.starts_with(&[0x89, 0x50, 0x4E, 0x47]), "not a PNG");

    let _ = std::fs::remove_dir_all(&dir);
}
