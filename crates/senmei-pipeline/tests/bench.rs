//! Real-GPU benchmarks for the selected upscaler at 1080p.
//! Run one at a time: cargo test -p senmei-pipeline --release --test bench -- --ignored --nocapture
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
/// (ROCm; needs the `tch` feature, and only exercises the non-fused path).
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
    let mut engine = senmei_ml::engine_for_model(&mref, senmei_ml::EngineBackend::default(), &std::env::temp_dir()).unwrap();
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
    let mut engine =
        senmei_ml::engine_for_model(&mref, backend(), &std::env::temp_dir()).unwrap();
    engine.load(&mref).unwrap();
    let mut frames = bench_frames();
    let total = frames.len();

    let mut step = senmei_pipeline::Upscale::new(bench_scale(), Some(engine));
    step.process(&mut frames[0]).unwrap(); // warm-up
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

/// App path with readback pipelining: 1-frame batches (BATCH_SIZE=1) through
/// `process_batch`, with `pipeline_depth` 1..3. Measures whether deferred
/// readbacks keep the GPU busier than the synchronous per-frame path.
#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscale_pipelined() {
    let _model_id =
        std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
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
    senmei_pipeline::set_pipeline_depth(1);
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

    let mut engine = senmei_ml::engine_for_model(&mref, senmei_ml::EngineBackend::default(), &std::env::temp_dir()).unwrap();
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

/// Real-frame upscaler sweep: every loadable `upscale` model at its native
/// scale on the given DVD frames. `BENCH_FRAMES` = comma-separated PNG paths
/// (default: the two frames in `models.bat/`). Measures the whole `Upscale`
/// step (frame → tiled infer → RGB8 frame), the app's render path.
/// Run: cargo test -p senmei-pipeline --release --test bench -- --ignored --nocapture bench_upscalers_real_frames
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
        )
        .unwrap();
        while let Some(f) = dec.next_frame().unwrap() {
            frames.push(f);
        }
    }
    let (w, h) = (frames[0].width, frames[0].height);
    // Save each model's upscaled output as `<id>.png` next to the inputs.
    let out_dir = std::path::Path::new(frames_env.split(',').next().unwrap().trim())
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    println!("\n==== real-frame upscaler sweep @ {w}x{h} ====");
    println!("outputs -> {}", out_dir.display());
    println!("{:<30} {:>5} {:>10} {:>8}", "model", "scale", "ms/frame", "FPS");

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
            let mut engine =
                senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
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
                let out = timed.last().unwrap();
                let bytes =
                    senmei_media::encode_png(out.width as u32, out.height as u32, &out.data)
                        .unwrap();
                std::fs::write(out_dir.join(format!("{id}.png")), bytes).unwrap();
                format!("{:>10.1} {:>8.1}", el * 1000.0, 1.0 / el)
            } else {
                let mut engine =
                    senmei_ml::engine_for_model(&mref, backend(), &dir).unwrap();
                engine.load(&mref).unwrap();
                let input = senmei_pipeline::frame_to_tensor(&frames[0]);
                let opts = InferOptions {
                    tile_size: Some(512),
                };
                let _ = senmei_ml::infer_tiled(engine.as_mut(), &input, &opts).unwrap();
                let t0 = Instant::now();
                let mut last = None;
                let mut iters = 0u32;
                for _ in 0..3 {
                    last = Some(senmei_ml::infer_tiled(engine.as_mut(), &input, &opts).unwrap());
                    iters += 1;
                }
                let out = senmei_pipeline::tensor_to_frame(
                    last.as_ref().unwrap(),
                    (frames[0].width as usize * mref.scale as usize) as u32,
                    (frames[0].height as usize * mref.scale as usize) as u32,
                );
                let bytes = senmei_media::encode_png(out.width as u32, out.height as u32, &out.data)
                    .unwrap();
                std::fs::write(out_dir.join(format!("{id}.png")), bytes).unwrap();
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
