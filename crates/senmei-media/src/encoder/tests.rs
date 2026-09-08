//! Encoder tests.

use super::*;
use std::process::Command;

fn untagged_pal_dvd() -> crate::VideoInfo {
    crate::VideoInfo {
        width: 720,
        height: 576,
        fps: 25.0,
        duration: 1.0,
        video_duration: 1.0,
        rotation: 0,
        color_transfer: None,
        color_primaries: None,
        video_codec: Some("mpeg2video".into()),
        audio_codec: None,
        pix_fmt: Some("yuv420p".into()),
        par: Some("16:15".into()),
        dar: Some("4:3".into()),
        field_order: Some("tt".into()),
        audio_tracks: vec![],
        subtitle_tracks: vec![],
    }
}

#[test]
fn untagged_pal_dvd_gets_sdr_color_filter() {
    let source = untagged_pal_dvd();
    assert_eq!(pal_sdr_filter(&source, &[]), Some(PAL_SDR_FILTER));
    assert_eq!(
        pal_sdr_filter(&source, &["-color_trc".into(), "bt709".into()]),
        None
    );
}

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
fn copied_source_streams_tune_interleaving() {
    assert!(should_tune_interleaving(&[]));
    assert!(should_tune_interleaving(&[
        "-an".into(),
        "-c:s".into(),
        "copy".into(),
    ]));
    assert!(!should_tune_interleaving(&["-an".into()]));
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
        vaapi_device(false)
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
    // Valid input (2 s silent AAC) so the optional `-map 1:a?` + `-shortest`
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
            video_dur_ms: None,
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
                    video_dur_ms: None,
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

/// Regression: keeps every source audio stream (`-map 1:a?` — was `1:a:0?`).
#[test]
fn encode_keeps_all_audio_streams() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("two-audio.mkv");
    let out = dir.join("two-audio-out.mkv");
    for p in [&input, &out] {
        let _ = std::fs::remove_file(p);
    }
    // 1 s video + two AAC tracks; mpeg4 is LGPL-safe for the source encode.
    let make = Command::new(ff)
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25",
            "-t",
            "1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-metadata:s:a:0",
            "language=ger",
            "-metadata:s:a:1",
            "language=eng",
            "-shortest",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(make.success(), "failed to create two-audio input");
    let mut enc = Encoder::open(
        &EncodeOptions {
            ffmpeg: ff,
            input: &input,
            output: &out,
            width: 64,
            height: 64,
            fps: 25.0,
            start_ms: 0,
            duration_ms: None,
            video_dur_ms: None,
        },
        &["-c:a".to_string(), "copy".to_string()],
    )
    .unwrap();
    let frame = Frame {
        width: 64,
        height: 64,
        data: vec![0u8; 64 * 64 * 3],
    };
    for _ in 0..25 {
        enc.write_frame(&frame).unwrap();
    }
    enc.finish().unwrap();
    let info = crate::probe::probe(&crate::ffmpeg::ffprobe_next_to(ff), &out).unwrap();
    assert_eq!(
        info.audio_tracks.len(),
        2,
        "render must keep every source audio stream"
    );
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&out);
}

/// Regression: feeding past the estimated `video_dur_ms` must not delete the file.
#[test]
fn encode_survives_feeding_past_video_dur() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("dur-input.mp4");
    let out = dir.join("dur-out.mkv");
    for p in [&input, &out] {
        let _ = std::fs::remove_file(p);
    }
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
    assert!(make.success(), "failed to create audio input");
    // Feed 2 s while video_dur_ms says 1 s (the mismatch that used to EPIPE).
    let mut enc = Encoder::open(
        &EncodeOptions {
            ffmpeg: ff,
            input: &input,
            output: &out,
            width: 64,
            height: 64,
            fps: 25.0,
            start_ms: 0,
            duration_ms: None,
            video_dur_ms: Some(1000),
        },
        &["-c:a".to_string(), "copy".to_string()],
    )
    .unwrap();
    let frame = Frame {
        width: 64,
        height: 64,
        data: vec![0u8; 64 * 64 * 3],
    };
    for _ in 0..50 {
        enc.write_frame(&frame).unwrap();
    }
    enc.finish().unwrap();
    assert!(
        out.exists() && out.metadata().unwrap().len() > 0,
        "file must survive a video_dur_ms mismatch"
    );
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&out);
}

/// Regression: range temp keeps every audio stream too (`0:a?` — was `0:a:0?`).
#[test]
fn extract_audio_range_keeps_all_tracks() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("two-audio-range.mkv");
    let _ = std::fs::remove_file(&input);
    let make = Command::new(ff)
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25",
            "-t",
            "2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=2",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(make.success(), "failed to create two-audio input");
    let tmp = extract_audio_range(ff, &input, 500, Some(1000)).expect("temp extraction");
    // `probe()` needs a video stream, but the temp is audio-only — count via ffprobe.
    let out = Command::new(crate::ffmpeg::ffprobe_next_to(ff))
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ])
        .arg(&tmp)
        .output()
        .unwrap();
    assert!(out.status.success(), "ffprobe failed on temp audio");
    let audio = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.trim() == "audio")
        .count();
    assert_eq!(audio, 2, "temp audio must keep every source audio stream");
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&input);
}

/// MKV source with two SRT subtitle streams (ger/eng) for mapping tests.
fn make_subtitle_source(ff: &Path, out: &Path) {
    let dir = out.parent().unwrap();
    let ger = dir.join("ger.srt");
    let eng = dir.join("eng.srt");
    std::fs::write(&ger, "1\n00:00:00,000 --> 00:00:01,000\nGER\n").unwrap();
    std::fs::write(&eng, "1\n00:00:00,000 --> 00:00:01,000\nENG\n").unwrap();
    let make = Command::new(ff)
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25",
            "-t",
            "1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
        ])
        .arg("-i")
        .arg(&ger)
        .arg("-i")
        .arg(&eng)
        .args([
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2",
            "-map",
            "3",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-c:s",
            "srt",
            "-metadata:s:s:0",
            "language=ger",
            "-metadata:s:s:1",
            "language=eng",
            "-shortest",
        ])
        .arg(out)
        .status()
        .unwrap();
    assert!(make.success(), "failed to create subtitle source");
    let _ = std::fs::remove_file(&ger);
    let _ = std::fs::remove_file(&eng);
}

/// Encode one second of frames and return the output's subtitle languages.
fn encode_subtitle_langs(ff: &Path, input: &Path, out: &Path, extra: &[&str]) -> Vec<String> {
    let extra: Vec<String> = extra.iter().map(|s| s.to_string()).collect();
    let mut enc = Encoder::open(
        &EncodeOptions {
            ffmpeg: ff,
            input,
            output: out,
            width: 64,
            height: 64,
            fps: 25.0,
            start_ms: 0,
            duration_ms: None,
            video_dur_ms: None,
        },
        &extra,
    )
    .unwrap();
    let frame = Frame {
        width: 64,
        height: 64,
        data: vec![0u8; 64 * 64 * 3],
    };
    for _ in 0..25 {
        enc.write_frame(&frame).unwrap();
    }
    enc.finish().unwrap();
    let probe = Command::new(crate::ffmpeg::ffprobe_next_to(ff))
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream_tags=language",
            "-of",
            "csv=p=0",
        ])
        .arg(out)
        .output()
        .unwrap();
    assert!(probe.status.success(), "ffprobe failed on subtitle output");
    String::from_utf8_lossy(&probe.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Regression: `-c:s copy` maps all subs; an explicit `-map 1:s:…` keeps the pick.
#[test]
fn encode_subtitle_copy_all_or_pick() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("two-sub.mkv");
    let all = dir.join("sub-all.mkv");
    let pick = dir.join("sub-pick.mkv");
    for p in [&input, &all, &pick] {
        let _ = std::fs::remove_file(p);
    }
    make_subtitle_source(ff, &input);
    let langs_all = encode_subtitle_langs(ff, &input, &all, &["-c:a", "copy", "-c:s", "copy"]);
    assert_eq!(
        langs_all,
        vec!["ger", "eng"],
        "-c:s copy must map all subtitles"
    );
    let langs_pick = encode_subtitle_langs(
        ff,
        &input,
        &pick,
        &["-c:a", "copy", "-c:s", "copy", "-map", "1:s:1"],
    );
    assert_eq!(
        langs_pick,
        vec!["eng"],
        "explicit -map must keep only the pick"
    );
    for p in [&input, &all, &pick] {
        let _ = std::fs::remove_file(p);
    }
}

/// Regression: ranged renders keep subtitles (temp had audio only before).
#[test]
fn encode_range_keeps_subtitles() {
    let Some(ff) = std::env::var("SENMEI_FFMPEG")
        .ok()
        .filter(|p| !p.is_empty())
    else {
        eprintln!("SENMEI_FFMPEG not set, skipping");
        return;
    };
    let ff = Path::new(&ff);
    let dir = std::env::temp_dir().join("senmei-enc-test");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("range-sub.mkv");
    let out = dir.join("range-sub-out.mkv");
    for p in [&input, &out] {
        let _ = std::fs::remove_file(p);
    }
    make_subtitle_source(ff, &input);
    let extra: Vec<String> = ["-c:a", "copy", "-c:s", "copy"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut enc = Encoder::open(
        &EncodeOptions {
            ffmpeg: ff,
            input: &input,
            output: &out,
            width: 64,
            height: 64,
            fps: 25.0,
            start_ms: 0,
            duration_ms: Some(1000),
            video_dur_ms: Some(1000),
        },
        &extra,
    )
    .unwrap();
    let frame = Frame {
        width: 64,
        height: 64,
        data: vec![0u8; 64 * 64 * 3],
    };
    for _ in 0..25 {
        enc.write_frame(&frame).unwrap();
    }
    enc.finish().unwrap();
    let probe = Command::new(crate::ffmpeg::ffprobe_next_to(ff))
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream_tags=language",
            "-of",
            "csv=p=0",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "ffprobe failed on ranged subtitle output"
    );
    let subs: Vec<String> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        subs,
        vec!["ger", "eng"],
        "ranged render must keep subtitles"
    );
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn parse_sentinels_hw_value() {
    let mut args = vec![
        "-senmei_encoder".into(),
        "hw".into(),
        "-c:v".into(),
        "copy".into(),
    ];
    let (pref, _, _) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Hardware));
    assert_eq!(args, vec!["-c:v", "copy"]);
}

#[test]
fn parse_sentinels_sw_value() {
    let mut args = vec!["-senmei_encoder".into(), "sw".into()];
    let (pref, _, _) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Software));
    assert!(args.is_empty());
}

#[test]
fn parse_sentinels_unknown_value() {
    let mut args = vec!["-senmei_encoder".into(), "bogus".into()];
    let (pref, _, _) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Auto));
    assert!(args.is_empty());
}

#[test]
fn parse_sentinels_missing_value() {
    let mut args = vec!["-senmei_encoder".into()];
    let (pref, _, _) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Auto));
    assert!(
        args.is_empty(),
        "trailing sentinel without value should be removed"
    );
}

#[test]
fn parse_sentinels_no_sentinels() {
    let mut args = vec!["-c:v".into(), "copy".into()];
    let (pref, _, _) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Auto));
    assert_eq!(args, vec!["-c:v", "copy"]);
}

#[test]
fn parse_sentinels_vaapi_10bit_detected() {
    let mut args = vec!["-pix_fmt".into(), "yuv420p10le".into()];
    let (_, vaapi_10bit, _) = parse_sentinels(&mut args);
    assert!(vaapi_10bit);
}

#[test]
fn parse_sentinels_both_sentinels_removed() {
    let mut args = vec![
        "-senmei_encoder".into(),
        "hw".into(),
        "-senmei_vaapi".into(),
        "igpu".into(),
        "-c:v".into(),
        "copy".into(),
    ];
    let (pref, _, prefer_igpu) = parse_sentinels(&mut args);
    assert!(matches!(pref, EncoderPref::Hardware));
    assert!(prefer_igpu, "igpu sentinel must select the iGPU");
    assert_eq!(args, vec!["-c:v", "copy"]);
}

#[test]
fn parse_sentinels_prefer_igpu_is_per_call() {
    let mut igpu = vec!["-senmei_vaapi".into(), "igpu".into()];
    let (_, _, prefer_igpu) = parse_sentinels(&mut igpu);
    assert!(prefer_igpu);
    let mut auto = vec!["-senmei_vaapi".into(), "auto".into()];
    let (_, _, prefer_igpu) = parse_sentinels(&mut auto);
    assert!(!prefer_igpu, "auto must not inherit a previous igpu pick");
    let mut none = vec!["-c:v".into(), "copy".into()];
    let (_, _, prefer_igpu) = parse_sentinels(&mut none);
    assert!(
        !prefer_igpu,
        "missing sentinel defaults to the discrete GPU"
    );
}
