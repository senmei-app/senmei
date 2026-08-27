//! Real-GPU benchmarks for the selected upscaler at 1080p.
//! Run one at a time: cargo bench -p senmei-pipeline -- --ignored --nocapture
//! Model selectable via BENCH_MODEL (default: real-cugan-x2).
//!
//! `bench_upscaler_1080p_fullframe` measures the raw tiled `engine.infer`
//! path; `bench_upscale_step` measures the whole `Upscale` step end to end
//! (frame → tiles → RGB8 frame). Both use tiled inference — the fused
//! full-frame path was removed because it OOMs autotune on large matmuls
//! (see docs/upstream-issues.md).

use std::time::Instant;

use senmei_ml::InferOptions;
use senmei_pipeline::Step;

/// Resolve `BENCH_MODEL` (or real-cugan-x2) from the registry.
fn model() -> senmei_ml::ModelRef {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let models_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry
        .resolve(&model_id, &models_dir)
        .expect("model in registry");
    let bpk = models_dir.join(&mref.path);
    assert!(bpk.exists(), "missing {bpk:?}");
    mref
}

/// Resolve `BENCH_BACKEND`: `vulkan` (default, burn) or `tch`/`libtorch`
/// (ROCm; enable via `--features tch`, and only exercises the non-fused path).
fn backend() -> senmei_ml::EngineBackend {
    match std::env::var("BENCH_BACKEND").as_deref() {
        Ok("tch") | Ok("libtorch") | Ok("rocm") => senmei_ml::EngineBackend::LibTorch,
        _ => senmei_ml::EngineBackend::Vulkan,
    }
}

/// Resolve `BENCH_SCALE` (requested upscale factor; default 2 — the model's
/// native scale, so the fused path runs without re-sampling).
fn bench_scale() -> u32 {
    std::env::var("BENCH_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

/// Generate a 2s 1080p24 testsrc and decode it to frames.
fn bench_frames() -> Vec<senmei_media::Frame> {
    let dir = std::env::temp_dir().join("senmei-bench-1080p");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=1920x1080:rate=24",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "failed to generate 1080p input");
    let ffmpeg = senmei_media::resolve(&dir);
    let mut dec = senmei_media::Decoder::open_with_range(
        &ffmpeg,
        &input,
        0,
        None,
        senmei_media::Tonemap::Auto,
        None,
    )
    .unwrap();
    let mut frames = Vec::new();
    while let Some(f) = dec.next_frame().unwrap() {
        frames.push(f);
    }
    assert!(!frames.is_empty(), "no frames decoded");
    frames
}

/// Plain tiled `engine.infer` throughput (convert-in + infer + convert-out).
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscaler_1080p_fullframe() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let mref = model();
    let mut engine = senmei_ml::engine_for_model(
        &mref,
        senmei_ml::EngineBackend::default(),
        &std::env::temp_dir(),
    )
    .unwrap();
    engine.load(&mref).unwrap();
    let mut frames = bench_frames();
    let total = frames.len();

    let opts = InferOptions {
        tile_size: Some(512),
    };
    let mut t_in = 0f64;
    let mut t_infer = 0f64;
    let mut t_out = 0f64;
    for (i, f) in frames.iter_mut().enumerate() {
        let t0 = Instant::now();
        let x = senmei_pipeline::frame_to_tensor(f);
        let t1 = Instant::now();
        let out = senmei_ml::infer_tiled(engine.as_mut(), &x, &opts).unwrap();
        let t2 = Instant::now();
        let _frame =
            senmei_pipeline::tensor_to_frame(&out, out.shape[3] as u32, out.shape[2] as u32);
        let t3 = Instant::now();
        if i == 0 {
            continue; // warm-up
        }
        t_in += (t1 - t0).as_secs_f64();
        t_infer += (t2 - t1).as_secs_f64();
        t_out += (t3 - t2).as_secs_f64();
    }

    let n = (total - 1) as f64;
    let ms_in = t_in * 1000.0 / n;
    let ms_infer = t_infer * 1000.0 / n;
    let ms_out = t_out * 1000.0 / n;
    let total_ms = ms_in + ms_infer + ms_out;
    println!("\n==== {model_id} 1080p -> 2160p tiled infer ====");
    println!("frames: {total} | convert-in {ms_in:.1} ms | infer {ms_infer:.1} ms | convert-out {ms_out:.1} ms");
    println!(
        "total {total_ms:.1} ms/frame | {:.1} FPS",
        1000.0 / total_ms
    );
    println!("=================================================");
}

/// Whole `Upscale` step throughput (frame → tiled infer → RGB8 frame) — the
/// app's render path.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscale_step() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let mref = model();
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).unwrap();
    engine.load(&mref).unwrap();
    let mut frames = bench_frames();
    let total = frames.len();

    let mut step = senmei_pipeline::Upscale::new(bench_scale(), Some(engine));
    let mut warm = frames[0].clone();
    step.process(&mut warm).unwrap(); // warm-up (on a clone — process rewrites the frame to the upscaled size)
    let s0 = Instant::now();
    for f in &mut frames {
        step.process(f).unwrap();
    }
    let s_el = s0.elapsed().as_secs_f64() / total as f64;
    println!("\n==== {model_id} Upscale step (tiled, app path) ====");
    println!(
        "frames: {total} | scale {} | step {:.1} ms/frame | {:.1} FPS",
        bench_scale(),
        s_el * 1000.0,
        1.0 / s_el
    );
    println!("=================================================");
}

/// Per-frame vs fused `process_batch` throughput (the app's batch path uses
/// `BATCH_SIZE = 4`). Batch sizes 2/4/8 shown so the sweet spot is visible.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscale_batch() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let mref = model();
    let frames = bench_frames();
    let total = frames.len();
    let scale = bench_scale();

    // Per-frame fused single-RGB8 path — reference.
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).unwrap();
    engine.load(&mref).unwrap();
    let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));
    let mut fs = frames.clone();
    step.process(&mut fs[0]).unwrap(); // warm-up
    let t0 = Instant::now();
    for f in &mut fs {
        step.process(f).unwrap();
    }
    let single_ms = t0.elapsed().as_secs_f64() * 1000.0 / total as f64;
    drop(step);

    println!("\n==== {model_id} Upscale step: per-frame vs process_batch (scale {scale}) ====");
    println!(
        "frames: {total} | per-frame {single_ms:.1} ms/frame ({:.1} FPS)",
        1000.0 / single_ms
    );
    for batch in [2usize, 4, 8] {
        let mut engine =
            senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).unwrap();
        engine.load(&mref).unwrap();
        let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));
        let mut fb = frames.clone();
        let mut warm = fb[..batch.min(fb.len())].to_vec();
        step.process_batch(&mut warm).unwrap(); // warm-up
        let t0 = Instant::now();
        for chunk in fb.chunks_mut(batch) {
            let mut v = chunk.to_vec();
            step.process_batch(&mut v).unwrap();
            chunk.clone_from_slice(&v);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / total as f64;
        println!(
            "batch {batch}: {ms:.1} ms/frame ({:.1} FPS) | vs per-frame {:.0}%",
            1000.0 / ms,
            ms * 100.0 / single_ms
        );
    }
    println!("===================================================");
}

/// Multi-frame batching on the REAL DVD frame (the app's 576×432 path), to
/// see whether MIOpen (tch) amortizes batched convs where Vulkan (burn)
/// regresses. `BENCH_MODEL` / `BENCH_SCALE` / `BENCH_FRAME` select the run.
/// The single decoded frame is replicated `REP` times so batches fill.
#[test]
#[ignore = "benchmark: requires GPU + model bpk + ffmpeg"]
fn bench_upscale_batch_dvd() {
    let mref = model();
    let scale: u32 = std::env::var("BENCH_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let rep: usize = std::env::var("BENCH_REP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let frame_path = std::env::var("BENCH_FRAME").unwrap_or_else(|_| {
        format!(
            "{}/models.bat/vlcsnap-2026-08-24-20h04m58s914.png",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../..")
        )
    });
    let dir = std::env::temp_dir().join("senmei-bench-dvd");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = senmei_media::resolve(&dir);
    let mut dec = senmei_media::Decoder::open_with_range(
        &ffmpeg,
        std::path::Path::new(&frame_path),
        0,
        None,
        senmei_media::Tonemap::Auto,
        None,
    )
    .unwrap();
    let first = dec.next_frame().unwrap().expect("frame");
    let frames: Vec<_> = (0..rep).map(|_| first.clone()).collect();
    let total = frames.len();

    // Reference: 1-frame batches with the app's pipelined depth (the real
    // app path) — the fair baseline for whether larger batches help MIOpen.
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
    engine.load(&mref).unwrap();
    senmei_pipeline::set_pipeline_depth(2);
    let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));
    let mut first = vec![frames[0].clone()];
    step.process_batch(&mut first).unwrap();
    let t0 = Instant::now();
    let mut out_n = 0usize;
    for f in &frames[1..] {
        let mut v = vec![f.clone()];
        step.process_batch(&mut v).unwrap();
        out_n += v.len();
    }
    let mut tail = Vec::new();
    step.flush(&mut tail).unwrap();
    out_n += tail.len();
    let single_ms = t0.elapsed().as_secs_f64() * 1000.0 / (total - 1) as f64;
    drop(step);
    senmei_pipeline::set_pipeline_depth(0);
    assert_eq!(out_n, total - 1, "reference must produce all frames");

    println!(
        "\n==== {} DVD-frames: per-frame vs process_batch (scale {scale}) ====",
        mref.id
    );
    println!(
        "frames: {total} (rep {rep}) | per-frame(pipelined) {single_ms:.1} ms/frame ({:.1} FPS)",
        1000.0 / single_ms
    );
    for batch in [1usize, 2, 4, 8] {
        let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
        engine.load(&mref).unwrap();
        let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));
        let fb = frames.clone();
        let mut warm = fb[..batch.min(fb.len())].to_vec();
        step.process_batch(&mut warm).unwrap(); // warm-up
        let t0 = Instant::now();
        let mut out_n = 0usize;
        for chunk in fb.chunks(batch) {
            let mut v = chunk.to_vec();
            step.process_batch(&mut v).unwrap();
            out_n += v.len(); // deferred path may hold frames in flight
        }
        let mut tail = Vec::new();
        step.flush(&mut tail).unwrap();
        out_n += tail.len();
        assert_eq!(out_n, total, "batch {batch} must produce all frames");
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / total as f64;
        println!(
            "batch {batch}: {ms:.1} ms/frame ({:.1} FPS) | vs per-frame {:.0}%",
            1000.0 / ms,
            ms * 100.0 / single_ms
        );
    }
    println!("===================================================");
}

/// App path with readback pipelining: 1-frame batches (BATCH_SIZE=1) through
/// `process_batch`, with `pipeline_depth` 1..3. Measures whether deferred
/// readbacks keep the GPU busier than the synchronous per-frame path.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscale_pipelined() {
    let _model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let mref = model();
    let frames = bench_frames();
    let total = frames.len();
    let scale = bench_scale();

    for depth in [1usize, 2, 3] {
        let mut engine =
            senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).unwrap();
        engine.load(&mref).unwrap();
        senmei_pipeline::set_pipeline_depth(depth);
        let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));

        // First frame fixes the dims (resolves synchronously).
        let mut first = vec![frames[0].clone()];
        step.process_batch(&mut first).unwrap();
        assert_eq!(first.len(), 1, "first batch must resolve synchronously");

        let t0 = Instant::now();
        let mut out_n = 0usize;
        for f in &frames[1..] {
            let mut batch = vec![f.clone()];
            step.process_batch(&mut batch).unwrap();
            out_n += batch.len();
        }
        let mut tail = Vec::new();
        step.flush(&mut tail).unwrap();
        out_n += tail.len();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / (total - 1) as f64;
        assert_eq!(out_n, total - 1, "deferred frames must all come out");
        println!(
            "pipelined depth {depth}: {ms:.1} ms/frame ({:.1} FPS) | out {out_n}",
            1000.0 / ms
        );
    }
    senmei_pipeline::set_pipeline_depth(0); // back to the owning default
    println!("===================================================");
}

/// Fused RGB8 path at a REQUESTED scale on the real DVD frame (the app path
/// for e.g. a 2× model rendered at 4×) — measures native-canvas accumulation
/// + the single final re-sample. `BENCH_MODEL` / `BENCH_SCALE` /
/// `BENCH_FRAME` select the run.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_fused_requested_scale() {
    let mref = model();
    let scale: u32 = std::env::var("BENCH_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let frame_path = std::env::var("BENCH_FRAME").unwrap_or_else(|_| {
        format!(
            "{}/models.bat/vlcsnap-2026-08-24-20h04m58s914.png",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../..")
        )
    });
    let dir = std::env::temp_dir().join("senmei-bench-fused");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = senmei_media::resolve(&dir);
    let mut dec = senmei_media::Decoder::open_with_range(
        &ffmpeg,
        std::path::Path::new(&frame_path),
        0,
        None,
        senmei_media::Tonemap::Auto,
        None,
    )
    .unwrap();
    let f = dec.next_frame().unwrap().expect("frame");
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
    engine.load(&mref).unwrap();
    let mut step = senmei_pipeline::Upscale::new(scale, Some(engine));
    let mut warm = vec![f.clone()];
    step.process_batch(&mut warm).unwrap(); // first batch resolves synchronously
    let (w, h) = (warm[0].width, warm[0].height);
    let n = 4;
    let t0 = Instant::now();
    let mut out_n = 0usize;
    for _ in 0..n {
        let mut batch = vec![f.clone()];
        step.process_batch(&mut batch).unwrap();
        out_n += batch.len();
    }
    let mut tail = Vec::new();
    step.flush(&mut tail).unwrap();
    out_n += tail.len();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!(
        "{:<30} fused @ s{scale}: {w}x{h} {ms:.1} ms ({fps:.1} FPS) | out {out_n}",
        mref.id,
        ms = ms,
        fps = 1000.0 / ms
    );
    println!("===================================================");
}

/// End-to-end render speed through the (threaded) pipeline, incl. x264 encode.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_pipeline_full_render() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let mref = model();

    let dir = std::env::temp_dir().join("senmei-bench-render");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let output = dir.join("output.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=1920x1080:rate=24",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok);

    let mut engine = senmei_ml::engine_for_model(
        &mref,
        senmei_ml::EngineBackend::default(),
        &std::env::temp_dir(),
    )
    .unwrap();
    engine.load(&mref).unwrap();
    let steps: Vec<Box<dyn senmei_pipeline::Step>> =
        vec![Box::new(senmei_pipeline::Upscale::new(2, Some(engine)))];
    let mut pipeline = senmei_pipeline::Pipeline::new(steps);

    let ffmpeg = senmei_media::resolve(&dir);
    let frames = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counter = frames.clone();
    let start = Instant::now();
    pipeline
        .run(&ffmpeg, &input, &output, move |p| {
            counter.store(p.frames_processed, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let n = frames.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "full pipeline ({model_id} 1080p->2160p, threaded + encode): {n} frames in {elapsed:.1}s -> {:.1} FPS",
        n as f64 / elapsed
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Render one model at a REQUESTED scale (native or re-sampled) on a real
/// frame and save the PNG next to it. `BENCH_MODEL` / `BENCH_SCALE` /
/// `BENCH_FRAME` select the run — e.g. a 2× model rendered at 4× shows how the
/// bilinear re-sample preserves grain vs a native 4× model.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscaler_requested_scale_png() {
    let scale: u32 = std::env::var("BENCH_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let frame_path = std::env::var("BENCH_FRAME").unwrap_or_else(|_| {
        format!(
            "{}/models.bat/vlcsnap-2026-08-24-20h04m58s914.png",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../..")
        )
    });
    let mref = model();
    let dir = std::env::temp_dir().join("senmei-bench-scale");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = senmei_media::resolve(&dir);
    let mut dec = senmei_media::Decoder::open_with_range(
        &ffmpeg,
        std::path::Path::new(&frame_path),
        0,
        None,
        senmei_media::Tonemap::Auto,
        None,
    )
    .unwrap();
    let f = dec.next_frame().unwrap().expect("frame");
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
    engine.load(&mref).unwrap();
    // The fused RGB8 path rejects 4× (VRAM guard), so run tiled at the model's
    // native scale, then re-sample to the requested scale via the Resize step
    // (the app path for e.g. a 2× model rendered at 4×).
    let input = senmei_pipeline::frame_to_tensor(&f);
    let opts = InferOptions {
        tile_size: Some(512),
    };
    let native = mref.scale as usize;
    let render = |engine: &mut dyn senmei_ml::InferenceEngine| {
        let out = senmei_ml::infer_tiled(engine, &input, &opts).expect("tiled infer");
        let mut frame = senmei_pipeline::tensor_to_frame(
            &out,
            (f.width as usize * native) as u32,
            (f.height as usize * native) as u32,
        );
        let factor = scale as f32 / native as f32;
        if (factor - 1.0).abs() > 1e-6 {
            senmei_pipeline::Resize::new(factor)
                .process(&mut frame)
                .expect("resize to requested scale");
        }
        frame
    };
    let mut frame = render(engine.as_mut());
    let t0 = Instant::now();
    for _ in 0..3 {
        frame = render(engine.as_mut());
    }
    let el = t0.elapsed().as_secs_f64() / 3.0;
    let bytes =
        senmei_media::encode_png(frame.width as u32, frame.height as u32, &frame.data).unwrap();
    let out_path = std::path::Path::new(&frame_path)
        .parent()
        .unwrap()
        .join(format!("{}__s{scale}.png", mref.id));
    std::fs::write(&out_path, bytes).unwrap();
    println!(
        "{:<30} @ s{scale}: {}x{} {:.1} ms ({:.1} FPS)",
        mref.id,
        frame.width,
        frame.height,
        el * 1000.0,
        1.0 / el
    );
}

/// Real-frame upscaler sweep: every loadable `upscale` model at its native
/// scale on the given DVD frames. `BENCH_FRAMES` = comma-separated PNG paths
/// (default: the two frames in `models.bat/`). Measures the whole `Upscale`
/// step (frame → tiled infer → RGB8 frame), the app's render path.
/// Run: cargo bench -p senmei-pipeline -- --ignored --nocapture bench_upscalers_real_frames
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscalers_real_frames() {
    let frames_env = std::env::var("BENCH_FRAMES").unwrap_or_else(|_| {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        format!(
            "{root}/models.bat/frame_cb6053b840e2b5c5.png,{root}/models.bat/frame_cb6053b840e2b5c5 (2).png"
        )
    });
    let models_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let upscale_ids: Vec<String> = registry
        .models()
        .iter()
        .filter(|m| matches!(m.kind, senmei_ml::ModelKind::Upscale) && m.loadable)
        .map(|m| m.id.clone())
        .collect();

    // Decode the PNGs to frames (ffmpeg handles single images).
    let dir = std::env::temp_dir().join("senmei-bench-frames");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = senmei_media::resolve(&dir);
    let mut frames: Vec<senmei_media::Frame> = Vec::new();
    for png in frames_env.split(',') {
        let mut dec = senmei_media::Decoder::open_with_range(
            &ffmpeg,
            std::path::Path::new(png.trim()),
            0,
            None,
            senmei_media::Tonemap::Auto,
            None,
        )
        .unwrap();
        while let Some(f) = dec.next_frame().unwrap() {
            frames.push(f);
        }
    }
    let (w, h) = (frames[0].width, frames[0].height);
    // One output PNG per input frame: `<id>__<source-stem>.png` next to the
    // inputs, so every image in `BENCH_FRAMES` gets its own upscaled result.
    let tags: Vec<String> = frames_env
        .split(',')
        .map(|p| {
            std::path::Path::new(p.trim())
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frame".into())
        })
        .collect();
    let out_dir = std::path::Path::new(frames_env.split(',').next().unwrap().trim())
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    println!("\n==== real-frame upscaler sweep @ {w}x{h} ====");
    println!("outputs -> {}", out_dir.display());
    println!(
        "{:<30} {:>5} {:>10} {:>8}",
        "model", "scale", "ms/frame", "FPS"
    );

    for id in &upscale_ids {
        let Some(mref) = registry.resolve(id, &models_dir) else {
            continue;
        };
        let bpk = models_dir.join(&mref.path);
        if !bpk.exists() {
            println!("{id:<30} -- no local bpk, skipped");
            continue;
        }
        // Panic-isolate each model: one broken model must not abort the sweep.
        let row = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
            engine.load(&mref).unwrap();
            let mut step = senmei_pipeline::Upscale::new(mref.scale, Some(engine));
            let mut warm = frames.clone();
            // Warm-up (loads weights, allocates). The fused RGB8 path trips
            // the VRAM guard on large outputs; fall back to raw tiled GPU
            // infer for those, the only viable path at this resolution.
            let fused = step.process(&mut warm[0]).is_ok();
            if fused {
                let mut timed = frames.clone();
                let t0 = Instant::now();
                let mut iters = 0usize;
                for f in &mut timed {
                    step.process(f).unwrap();
                    iters += 1;
                }
                let el = t0.elapsed().as_secs_f64() / iters as f64;
                for (i, out) in timed.iter().enumerate() {
                    let bytes =
                        senmei_media::encode_png(out.width as u32, out.height as u32, &out.data)
                            .unwrap();
                    std::fs::write(out_dir.join(format!("{id}__{}.png", tags[i])), bytes).unwrap();
                }
                format!("{:>10.1} {:>8.1}", el * 1000.0, 1.0 / el)
            } else {
                let mut engine = senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
                engine.load(&mref).unwrap();
                let opts = InferOptions {
                    tile_size: Some(512),
                };
                let warm = senmei_pipeline::frame_to_tensor(&frames[0]);
                let _ = senmei_ml::infer_tiled(engine.as_mut(), &warm, &opts).unwrap();
                let t0 = Instant::now();
                let mut iters = 0u32;
                for (i, f) in frames.iter().enumerate() {
                    let input = senmei_pipeline::frame_to_tensor(f);
                    let out = senmei_ml::infer_tiled(engine.as_mut(), &input, &opts).unwrap();
                    let frame = senmei_pipeline::tensor_to_frame(
                        &out,
                        (f.width as usize * mref.scale as usize) as u32,
                        (f.height as usize * mref.scale as usize) as u32,
                    );
                    let bytes = senmei_media::encode_png(
                        frame.width as u32,
                        frame.height as u32,
                        &frame.data,
                    )
                    .unwrap();
                    std::fs::write(out_dir.join(format!("{id}__{}.png", tags[i])), bytes).unwrap();
                    iters += 1;
                }
                let el = t0.elapsed().as_secs_f64() / iters as f64;
                format!("{:>10.1} {:>8.1}  *tiled", el * 1000.0, 1.0 / el)
            }
        }));
        match row {
            Ok(val) => println!("{:<30} {:>5} {val}", id, mref.scale),
            Err(p) => {
                let msg = p
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".into());
                println!("{:<30} {:>5}  PANIC: {msg}", id, mref.scale);
            }
        }
    }
    println!("  * fused RGB8 path exceeds the VRAM guard at this resolution; raw tiled GPU infer");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- Aux-stack benchmark (interp / denoise / deblur) ----

/// PSNR (dB) between two packed rgb24 buffers of the same length.
fn psnr_db(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr: size mismatch");
    let n = a.len() as f64;
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / n;
    if mse < 1e-12 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Mean SSIM over 8×8 windows (RGB channels averaged) between two rgb24 frames.
fn ssim_avg(a: &senmei_media::Frame, b: &senmei_media::Frame) -> f64 {
    debug_assert_eq!((a.width, a.height), (b.width, b.height));
    let w = a.width as usize;
    let h = a.height as usize;
    const K1: f64 = 0.01;
    const K2: f64 = 0.03;
    const C1: f64 = (K1 * 255.0) * (K1 * 255.0);
    const C2: f64 = (K2 * 255.0) * (K2 * 255.0);
    const WIN: usize = 8;
    let n = (WIN * WIN) as f64;
    let mut acc = 0.0f64;
    let mut count = 0u64;
    for c in 0..3 {
        let mut yy = 0;
        while yy + WIN <= h {
            let mut xx = 0;
            while xx + WIN <= w {
                let (mut ma, mut mb) = (0.0f64, 0.0f64);
                for y in yy..yy + WIN {
                    for x in xx..xx + WIN {
                        let p = (y * w + x) * 3 + c;
                        ma += a.data[p] as f64;
                        mb += b.data[p] as f64;
                    }
                }
                ma /= n;
                mb /= n;
                let (mut va, mut vb, mut cov) = (0.0f64, 0.0f64, 0.0f64);
                for y in yy..yy + WIN {
                    for x in xx..xx + WIN {
                        let p = (y * w + x) * 3 + c;
                        let da = a.data[p] as f64 - ma;
                        let db = b.data[p] as f64 - mb;
                        va += da * da;
                        vb += db * db;
                        cov += da * db;
                    }
                }
                va /= n;
                vb /= n;
                cov /= n;
                acc += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                    / ((ma * ma + mb * mb + C1) * (va + vb + C2));
                count += 1;
                xx += WIN;
            }
            yy += WIN;
        }
    }
    acc / count as f64
}

/// Deterministic xorshift64 — seeded so noise/blur are reproducible.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Box–Muller Gaussian (unit variance).
fn gauss(rng: &mut XorShift) -> f32 {
    let u1 = rng.unit().max(1e-9);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Add zero-mean Gaussian noise with std `sigma` (in 0..1) to a frame.
fn add_noise(frame: &senmei_media::Frame, sigma: f32) -> senmei_media::Frame {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let data = frame
        .data
        .iter()
        .map(|&v| {
            (v as f32 + gauss(&mut rng) * sigma * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect();
    senmei_media::Frame {
        width: frame.width,
        height: frame.height,
        data,
    }
}

/// Separable 5-tap Gaussian blur (σ≈1.5) on a frame.
fn gaussian_blur(frame: &senmei_media::Frame) -> senmei_media::Frame {
    const K: [f32; 5] = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136];
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut tmp = vec![0u8; frame.data.len()];
    let mut out = vec![0u8; frame.data.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, kw) in K.iter().enumerate() {
                    let xx = (x as isize + k as isize - 2).clamp(0, w as isize - 1) as usize;
                    acc += kw * frame.data[(y * w + xx) * 3 + c] as f32;
                }
                tmp[(y * w + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, kw) in K.iter().enumerate() {
                    let yy = (y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
                    acc += kw * tmp[(yy * w + x) * 3 + c] as f32;
                }
                out[(y * w + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    senmei_media::Frame {
        width: frame.width,
        height: frame.height,
        data: out,
    }
}

/// Load an aux-stack engine; `None` (with a note) when the weights are missing.
fn load_aux_engine(model_id: &str) -> Option<Box<dyn senmei_ml::InferenceEngine>> {
    let models_dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).ok()?;
    let mref = registry.resolve(model_id, &models_dir)?;
    if !models_dir.join(&mref.path).is_file() {
        eprintln!(
            "  (skip {model_id}: weights not downloaded — {})",
            mref.path.display()
        );
        return None;
    }
    let mut engine = senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).ok()?;
    engine.load(&mref).ok()?;
    Some(engine)
}

/// Aux-stack sweep: interpolation / denoise / deblur throughput + quality.
///
/// Every stack gets a quantitative quality score against a known reference
/// (PSNR dB + SSIM) plus the trivial no-op baseline it must beat:
/// - interp: drop the middle of a real triplet, interpolate it from its
///   neighbours, compare vs the real middle (baseline = linear blend);
/// - denoise: add Gaussian noise (σ=0.1) to a real frame, denoise, compare vs
///   clean (baseline = noisy input);
/// - deblur: Gaussian-blur a real frame, deblur, compare vs clean
///   (baseline = blurred input).
#[test]
#[ignore = "benchmark: requires Vulkan + aux model bpks + ffmpeg"]
fn bench_aux_stacks() {
    let dir = std::env::temp_dir().join("senmei-bench-aux");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = senmei_media::resolve(&dir);

    // --- Interpolation: consecutive testsrc2 frames (smooth motion) ---
    println!("\n==== Interpolation (720x576, factor 2, dropped-middle vs real) ====");
    let clip = dir.join("interp.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=2:size=720x576:rate=24",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
        ])
        .arg(&clip)
        .status()
        .unwrap()
        .success();
    assert!(ok, "failed to generate interp input");
    let mut dec = senmei_media::Decoder::open_with_range(
        &ffmpeg,
        &clip,
        0,
        None,
        senmei_media::Tonemap::Auto,
        None,
    )
    .unwrap();
    let mut frames = Vec::new();
    while let Some(f) = dec.next_frame().unwrap() {
        frames.push(f);
    }
    let opts = InferOptions { tile_size: None }; // interp is full-frame
                                                 // No-op baseline: linear blend of the neighbours (same for every model).
    let mut bps = 0.0f64;
    let mut bss = 0.0f64;
    let mut bn = 0u64;
    let mut i = 0;
    while i + 2 < frames.len() {
        let a_t = senmei_pipeline::frame_to_tensor(&frames[i]);
        let b_t = senmei_pipeline::frame_to_tensor(&frames[i + 2]);
        let real = &frames[i + 1];
        let bl = senmei_pipeline::tensor_to_frame(
            &senmei_ml::blend(&a_t, &b_t, 0.5),
            real.width,
            real.height,
        );
        bps += psnr_db(&bl.data, &real.data);
        bss += ssim_avg(&bl, real);
        bn += 1;
        i += 2;
    }
    println!(
        "| (linear blend, no model) | — | — | {:.1} | {:.3} |",
        bps / bn as f64,
        bss / bn as f64
    );
    for id in ["rife-v4.6", "ifrnet-vimeo90k", "ifrnet-gopro"] {
        let Some(mut engine) = load_aux_engine(id) else {
            continue;
        };
        let mut total = 0.0f64;
        let (mut ps, mut ss) = (0.0f64, 0.0f64);
        let mut n = 0u64;
        let mut i = 0;
        let mut first = true;
        while i + 2 < frames.len() {
            let a_t = senmei_pipeline::frame_to_tensor(&frames[i]);
            let b_t = senmei_pipeline::frame_to_tensor(&frames[i + 2]);
            let real = &frames[i + 1];
            let t0 = Instant::now();
            let out = engine
                .infer_interp(&a_t, &b_t, 0.5, &opts)
                .expect("model has no interp path")
                .unwrap();
            let dt = t0.elapsed().as_secs_f64();
            let mid = senmei_pipeline::tensor_to_frame(&out, real.width, real.height);
            if first {
                first = false; // warm-up (autotune / first-kernel)
            } else {
                total += dt;
                n += 1;
                ps += psnr_db(&mid.data, &real.data);
                ss += ssim_avg(&mid, real);
            }
            i += 2;
        }
        println!(
            "| {id} | {:.1} ms | {:.1} FPS | {:.1} | {:.3} |",
            total * 1000.0 / n as f64,
            n as f64 / total,
            ps / n as f64,
            ss / n as f64
        );
    }

    // --- Denoise / Deblur: real DVD frame + synthetic degradation ---
    let real_frame = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models.bat/vlcsnap-2026-08-24-20h04m58s914.png"
    ));
    let clean = if real_frame.is_file() {
        let mut dec = senmei_media::Decoder::open_with_range(
            &ffmpeg,
            &real_frame,
            0,
            None,
            senmei_media::Tonemap::Auto,
            None,
        )
        .unwrap();
        dec.next_frame().unwrap().expect("decode real frame")
    } else {
        eprintln!("  (real DVD frame missing; using a testsrc frame for denoise/deblur)");
        frames[0].clone()
    };
    let denoise_opts = InferOptions {
        tile_size: Some(640),
    };

    println!("\n==== Denoise (real frame + Gaussian noise σ=0.1) ====");
    let noisy = add_noise(&clean, 0.1);
    let bps = psnr_db(&noisy.data, &clean.data);
    let bss = ssim_avg(&noisy, &clean);
    println!("| (noisy input, no model) | — | — | {bps:.1} | {bss:.3} |");
    for id in [
        "drunet-color",
        "dncnn-color",
        "ffdnet-color",
        "scunet-denoise",
    ] {
        let Some(mut engine) = load_aux_engine(id) else {
            continue;
        };
        let noisy_t = senmei_pipeline::frame_to_tensor(&noisy);
        let _ =
            senmei_ml::infer_denoise_tiled(engine.as_mut(), &noisy_t, 0.1, &denoise_opts).unwrap(); // warm-up
        let t0 = Instant::now();
        let out =
            senmei_ml::infer_denoise_tiled(engine.as_mut(), &noisy_t, 0.1, &denoise_opts).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let den = senmei_pipeline::tensor_to_frame(&out, clean.width, clean.height);
        println!(
            "| {id} | {:.1} ms | {:.1} FPS | {:.1} | {:.3} |",
            dt * 1000.0,
            1.0 / dt,
            psnr_db(&den.data, &clean.data),
            ssim_avg(&den, &clean)
        );
    }

    println!("\n==== Deblur (real frame + Gaussian blur) ====");
    let blurred = gaussian_blur(&clean);
    let bps = psnr_db(&blurred.data, &clean.data);
    let bss = ssim_avg(&blurred, &clean);
    println!("| (blurred input, no model) | — | — | {bps:.1} | {bss:.3} |");
    for id in ["nafnet-gopro-width32"] {
        let Some(mut engine) = load_aux_engine(id) else {
            continue;
        };
        let blurred_t = senmei_pipeline::frame_to_tensor(&blurred);
        let full = InferOptions { tile_size: None }; // UNet: full-frame, no tiling
        let _ = senmei_ml::infer_tiled(engine.as_mut(), &blurred_t, &full).unwrap(); // warm-up
        let t0 = Instant::now();
        let out = senmei_ml::infer_tiled(engine.as_mut(), &blurred_t, &full).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let deb = senmei_pipeline::tensor_to_frame(&out, clean.width, clean.height);
        println!(
            "| {id} | {:.1} ms | {:.1} FPS | {:.1} | {:.3} |",
            dt * 1000.0,
            1.0 / dt,
            psnr_db(&deb.data, &clean.data),
            ssim_avg(&deb, &clean)
        );
    }
    println!("==========================================================");
    let _ = std::fs::remove_dir_all(&dir);
}
