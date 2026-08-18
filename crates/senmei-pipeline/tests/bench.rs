//! Real-GPU benchmark for the selected upscaler at 1080p.
//! Run with: cargo test -p senmei-pipeline --release --test bench -- --ignored --nocapture
//! Model selectable via BENCH_MODEL (default: real-cugan-x2).

use std::time::Instant;

use senmei_ml::InferOptions;
use senmei_pipeline::Step;

#[test]
#[ignore = "benchmark: requires Vulkan + model bpk + ffmpeg"]
fn bench_upscaler_1080p_fullframe() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let models_dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models"
    ));
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry.resolve(&model_id, &models_dir).expect("model in registry");
    let bpk = models_dir.join(mref.path);
    assert!(bpk.exists(), "missing {bpk:?}");

    let dir = std::env::temp_dir().join("senmei-bench-1080p");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=24",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok, "failed to generate 1080p input");

    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry.resolve(&model_id, &models_dir).unwrap();
    let mut engine = senmei_ml::engine_for_model(&mref).unwrap();
    engine.load(&mref).unwrap();

    let ffmpeg = senmei_media::resolve(&dir);
    let mut dec = senmei_media::Decoder::open_with_range(&ffmpeg, &input, 0, None).unwrap();
    let mut frames = Vec::new();
    while let Some(f) = dec.next_frame().unwrap() {
        frames.push(f);
    }
    let total = frames.len();
    assert!(total > 0, "no frames decoded");

    let opts = InferOptions { tile_size: Some(512) };
    let mut t_in = 0f64;
    let mut t_infer = 0f64;
    let mut t_out = 0f64;

    for (i, f) in frames.iter_mut().enumerate() {
        let t0 = Instant::now();
        let x = senmei_pipeline::frame_to_tensor(f);
        let t1 = Instant::now();
        let out = senmei_ml::infer_tiled(engine.as_mut(), &x, &opts).unwrap();
        let t2 = Instant::now();
        let _frame = senmei_pipeline::tensor_to_frame(
            &out,
            out.shape[3] as u32,
            out.shape[2] as u32,
        );
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
    println!("\n==== {model_id} 1080p -> 2160p full-frame ====");
    println!("frames: {total} | convert-in {ms_in:.1} ms | infer {ms_infer:.1} ms | convert-out {ms_out:.1} ms");
    println!("total {total_ms:.1} ms/frame | {:.1} FPS", 1000.0 / total_ms);
    println!("=================================================");

    // Real render step: now uses the fused f16->RGB8 path (no f32 tensor).
    let mut step = senmei_pipeline::Upscale::new(2, Some(engine));
    step.process(&mut frames[0]).unwrap(); // warm-up
    let s0 = Instant::now();
    for f in &mut frames {
        step.process(f).unwrap();
    }
    let s_el = s0.elapsed().as_secs_f64() / total as f64;
    println!(
        "step (fused RGB8): {:.1} ms/frame | {:.1} FPS",
        s_el * 1000.0,
        1.0 / s_el
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// End-to-end render speed through the (threaded) pipeline, incl. x264 encode.
#[test]
#[ignore = "benchmark: requires Vulkan + ShuffleCugan bpk + ffmpeg"]
fn bench_pipeline_full_render() {
    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "real-cugan-x2".to_string());
    let models_dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models"
    ));
    let mut registry = senmei_ml::Registry::new();
    registry.load_dir(&models_dir).unwrap();
    let mref = registry.resolve(&model_id, &models_dir).expect("model in registry");

    let dir = std::env::temp_dir().join("senmei-bench-render");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let output = dir.join("output.mp4");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=24",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
        ])
        .arg(&input)
        .status()
        .unwrap()
        .success();
    assert!(ok);

    let mut engine = senmei_ml::engine_for_model(&mref).unwrap();
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
        "full pipeline ({model_id} 1080p->2160p, threaded + x264): {n} frames in {elapsed:.1}s -> {:.1} FPS",
        n as f64 / elapsed
    );

    let _ = std::fs::remove_dir_all(&dir);
}
