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

/// HDR10 sources: probe detects PQ/bt2020 and the decoder applies the tonemap
/// filter (Auto) while Off skips it. Gated on libx265 (GPL-only, absent from
/// the LGPL portable build) because the clip needs a 10-bit encoder.
#[test]
fn hdr_source_is_detected_and_tonemapped() {
    let encoders = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output();
    let has_x265 = encoders
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("libx265"))
        .unwrap_or(false);
    if !has_x265 {
        eprintln!("libx265 not available, skipping HDR test");
        return;
    }

    let dir = std::env::temp_dir().join("senmei-hdr-test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input: PathBuf = dir.join("hdr.mp4");
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x180:rate=10",
            "-pix_fmt",
            "yuv420p10le",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
            "-tag:v",
            "hvc1",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "HDR generation failed");

    let info = senmei_media::probe(&input).expect("probe failed");
    assert!(info.is_hdr(), "HDR10 source should be detected as HDR");

    let ffmpeg = std::env::var("SENMEI_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let mut auto = senmei_media::Decoder::open_with_range(
        std::path::Path::new(&ffmpeg),
        &input,
        0,
        Some(1000),
        senmei_media::Tonemap::Auto,
    )
    .expect("auto decoder");
    let f = auto.next_frame().expect("next_frame").expect("a frame");
    assert_eq!((f.width, f.height), (320, 180));

    let mut off = senmei_media::Decoder::open_with_range(
        std::path::Path::new(&ffmpeg),
        &input,
        0,
        Some(1000),
        senmei_media::Tonemap::Off,
    )
    .expect("off decoder");
    off.next_frame().expect("next_frame").expect("a frame");

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
        senmei_media::Tonemap::Auto,
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
