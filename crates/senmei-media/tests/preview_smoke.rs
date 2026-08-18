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
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x180:rate=24",
            "-pix_fmt",
            "yuv420p",
        ])
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

/// Rotated videos: probe reports display dims + rotation, and the decoder
/// outputs display-sized frames (no silent autorotation mismatch). The rotation
/// is a real MP4 DisplayMatrix (ffmpeg autorotates it), and the decoded
/// content must match ffmpeg's own autorotation.
#[test]
fn probe_and_decode_apply_rotation() {
    let dir = std::env::temp_dir().join("senmei-rotation-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base: PathBuf = dir.join("base.mp4");
    let input: PathBuf = dir.join("rotated.mp4");
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x180:rate=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&base)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg base generation failed");
    let ok = Command::new("ffmpeg")
        .args(["-y", "-noautorotate", "-display_rotation", "90", "-i"])
        .arg(&base)
        .args(["-c", "copy"])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg rotated generation failed");

    let info = senmei_media::probe(&input).expect("probe failed");
    assert_eq!(info.rotation, 90, "rotation should be detected");
    assert_eq!(
        (info.width, info.height),
        (180, 320),
        "display dims should swap for 90°"
    );

    let ffmpeg = std::env::var("SENMEI_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let mut dec = senmei_media::Decoder::open_with_range(
        std::path::Path::new(&ffmpeg),
        &input,
        0,
        Some(2000),
    )
    .expect("decoder open");
    let frame = dec.next_frame().expect("next_frame").expect("a frame");
    assert_eq!(
        (frame.width, frame.height),
        (180, 320),
        "decoded frame should be display-sized"
    );

    // Content must match ffmpeg's own autorotation of the same file.
    let auto = Command::new(&ffmpeg)
        .args(["-y", "-i"])
        .arg(&input)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .unwrap()
        .stdout;
    assert_eq!(auto.len(), frame.data.len(), "same frame size");
    let diff = frame.data.iter().zip(&auto).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "decoded frame should equal ffmpeg autorotation");

    let _ = std::fs::remove_dir_all(&dir);
}
