//! Encoder tests.

use super::*;

#[test]
fn kvazaar_strips_tune() {
    let args = [
        "-tune".to_string(),
        "grain".to_string(),
        "-preset".to_string(),
        "medium".to_string(),
    ];
    assert_eq!(
        kvazaar_compat_args(&args),
        vec!["-preset".to_string(), "medium".to_string()]
    );
    let plain = ["-pix_fmt".to_string(), "yuv420p10le".to_string()];
    assert_eq!(kvazaar_compat_args(&plain), plain);
}

#[test]
fn vaapi_strips_software_encoder_flags() {
    let args = [
        "-preset".to_string(),
        "veryfast".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p10le".to_string(),
        "-tune".to_string(),
        "grain".to_string(),
        "-qp".to_string(),
        "18".to_string(),
    ];
    // Software flags are dropped, a caller-provided -qp passes through.
    assert_eq!(
        vaapi_compat_args(&args),
        vec!["-qp".to_string(), "18".to_string()]
    );
    let plain = ["-c:a".to_string(), "copy".to_string()];
    assert_eq!(vaapi_compat_args(&plain), plain);
}

#[test]
fn override_codec_sets_bitrate_for_openh264_only() {
    // libopenh264 is ABR-only: the override adds a resolution-based `-b:v`
    // unless the caller already passed one; other codecs get no defaults.
    let w = 1920u32;
    let h = 1080u32;
    let base = ["-c:v".into(), "libopenh264".into()];
    assert_eq!(
        override_codec_args("libopenh264", &base, w, h),
        vec!["-b:v".to_string(), "14400k".to_string()]
    );
    let with_bv = [
        "-c:v".into(),
        "libopenh264".into(),
        "-b:v".into(),
        "1000k".into(),
    ];
    assert_eq!(
        override_codec_args("libopenh264", &with_bv, w, h),
        Vec::<String>::new()
    );
    assert_eq!(
        override_codec_args("libkvazaar", &base, w, h),
        Vec::<String>::new()
    );
    assert_eq!(
        override_codec_args("libsvtav1", &base, w, h),
        Vec::<String>::new()
    );
}

/// Reproduce the app's real HW selection: real ffmpeg probes at the actual
/// output resolution, Hardware pref. Prints which codec gets chosen.
#[test]
fn probe_hw_selection() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let caps = crate::ffmpeg::probe(ff).encoders;
    let verify = hw_verifier(ff);
    let verify_full = |codec: &str| test_encode(ff, codec, 2304, 1728);
    println!(
        "caps has hevc_vaapi={} h264_vaapi={} | vaapi_device={:?}",
        caps.iter().any(|e| e == "hevc_vaapi"),
        caps.iter().any(|e| e == "h264_vaapi"),
        vaapi_device()
    );
    for codec in ["hevc_vaapi", "h264_vaapi"] {
        println!(
            "{codec}: verify(640)={} verify_full(2304x1728)={}",
            verify(codec),
            verify_full(codec)
        );
    }
    for pref in [
        EncoderPref::Auto,
        EncoderPref::Hardware,
        EncoderPref::Software,
    ] {
        let (codec, _) = pick_from_caps(&caps, 2304, 1728, pref, &verify, &verify_full);
        println!("SENMEI_FFMPEG probe @2304x1728 pref={pref:?} -> {codec}");
    }
}

#[test]
fn verified_hw_encoder_beats_software() {
    if HW_ENCODERS.is_empty() {
        return;
    }
    let mut caps = vec!["libkvazaar".to_string()];
    caps.extend(HW_ENCODERS.iter().map(|c| c.to_string()));
    let (codec, _) = pick_from_caps(
        &caps,
        1920,
        1080,
        EncoderPref::Auto,
        &|c| c == HW_ENCODERS[0],
        &|c| c == HW_ENCODERS[0],
    );
    assert_eq!(codec, HW_ENCODERS[0]);
}

#[test]
fn listed_but_unverified_hw_falls_back() {
    let mut caps = vec!["libkvazaar".to_string()];
    caps.extend(HW_ENCODERS.iter().map(|c| c.to_string()));
    let (codec, args) =
        pick_from_caps(&caps, 1920, 1080, EncoderPref::Auto, &|_| false, &|_| false);
    assert_eq!(codec, "libkvazaar");
    assert!(args.contains(&"-preset".to_string()));
}

#[test]
fn hevc_hw_comes_before_h264_hw() {
    if HW_ENCODERS.is_empty() {
        return;
    }
    assert!(
        HW_ENCODERS[0].starts_with("hevc_"),
        "HEVC first in {HW_ENCODERS:?}"
    );
    let caps: Vec<String> = HW_ENCODERS.iter().map(|c| c.to_string()).collect();
    let (codec, _) = pick_from_caps(&caps, 1920, 1080, EncoderPref::Auto, &|_| true, &|_| true);
    assert_eq!(codec, HW_ENCODERS[0]);
}

/// End-to-end encode through the selected (LGPL-safe) codec. Skipped unless
/// `SENMEI_FFMPEG` points at a real ffmpeg (e.g. the pinned BtbN LGPL build).
#[test]
fn encodes_through_selected_codec() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let (codec, _args) = pick_from_caps(
        &crate::ffmpeg::probe(ff).encoders,
        64,
        64,
        EncoderPref::Auto,
        &|_| false,
        &|_| false,
    );
    assert!(
        ["libkvazaar", "libopenh264", "libx264", "h264"].contains(&codec.as_str()),
        "unexpected codec {codec}"
    );

    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let out = dir.join("out.mp4");
    let _ = std::fs::remove_file(&out);
    // Valid input (2 s silent AAC) so the optional `-map 1:a:0?` + `-shortest`
    // don't kill the pipe: video (30 frames @30fps = 1 s) is the shortest.
    let make = Command::new(ff)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            "2",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "1",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(make.success(), "failed to create test input");
    let mut enc = Encoder::open(
        &EncodeOptions {
            ffmpeg: &ff,
            input: &input,
            output: &out,
            width: 64,
            height: 64,
            fps: 30.0,
            start_ms: 0,
            duration_ms: None,
        },
        &[],
    )
    .unwrap();
    let frame = Frame {
        width: 64,
        height: 64,
        data: vec![0u8; 64 * 64 * 3],
    };
    for _ in 0..30 {
        enc.write_frame(&frame).unwrap();
    }
    enc.finish().unwrap();
    assert!(out.exists() && out.metadata().unwrap().len() > 0);
    let status = Command::new(ff)
        .args(["-v", "error", "-i"])
        .arg(&out)
        .args(["-f", "null", "-"])
        .status()
        .unwrap();
    assert!(status.success(), "encoded output not decodable");
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&input);
}

/// Regression: ffmpeg's stderr is drained in the background, so an encode
/// that emits more than a 64-KiB pipe can hold still finishes. Without the
/// drain, `finish` deadlocks once the pipe is full (long-render hang).
#[test]
fn finish_after_stderr_overflows() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = PathBuf::from(ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("stderr-input.mp4");
    let out = dir.join("stderr-out.mp4");
    let _ = std::fs::remove_file(&out);
    // Audio longer than the video (10 s > 200 frames @30fps) so `-shortest`
    // doesn't end the pipe early and trip `write_frame` on a broken pipe.
    let make = Command::new(&ff)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            "10",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "1",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(make.success(), "failed to create test input");
    // `trace` makes ffmpeg emit far more stderr than the pipe can buffer.
    let extra = ["-loglevel".to_string(), "trace".to_string()];
    let (tx, rx) = std::sync::mpsc::channel();
    let input_t = input.clone();
    let out_t = out.clone();
    let _ = std::thread::spawn(move || {
        let run = (|| -> Result<()> {
            let mut enc = Encoder::open(
                &EncodeOptions {
                    ffmpeg: &ff,
                    input: &input_t,
                    output: &out_t,
                    width: 64,
                    height: 64,
                    fps: 30.0,
                    start_ms: 0,
                    duration_ms: None,
                },
                &extra,
            )?;
            let frame = Frame {
                width: 64,
                height: 64,
                data: vec![0u8; 64 * 64 * 3],
            };
            for _ in 0..200 {
                enc.write_frame(&frame)?;
            }
            enc.finish()
        })();
        let _ = tx.send(run);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("encode failed: {e}"),
        Err(_) => panic!("encode deadlocked on full stderr pipe"),
    }
    assert!(out.exists() && out.metadata().unwrap().len() > 0);
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&input);
}
