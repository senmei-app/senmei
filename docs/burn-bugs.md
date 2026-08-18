# burn (tracel-ai/burn) upstream bugs — findings

Bugs found while running `burn-wgpu` 0.21 / `cubecl` 0.10 on AMD RADV
(RX 9070 XT, RDNA4, `gfx1201`, non-conformant Vulkan). Kept here so each
finding can be turned into a proper upstream issue. Versions: `burn = 0.21.0`,
`burn-wgpu = 0.21.0` (features `vulkan`), `burn-fusion = 0.21.0`,
`cubecl = 0.10.0`.

---

## Bug 1 — burn-fusion: intermittent "Ordering is bigger than operations" panic

### Symptom

Deterministic *eventually*, nondeterministic *when*: a fused tensor readback
panics on the fusion server's worker thread (`DSU-0-0`) at
`burn-fusion/src/stream/execution/ordering.rs:49`:

```
thread 'DSU-0-0' panicked at .../burn-fusion-0.21.0/src/stream/execution/ordering.rs:49:13:
Ordering is bigger than operations
```

The caller thread then unwraps the resulting `CallError` at
`client.rs:175/189` (`read_tensor_int`/`read_tensor_float`) and panics too.
Happens with **any** model (Real-CUGAN, Fallin/UpCunet2xFast), **any** readback
dtype (f16, u8, f32), and **regardless** of whether the conversion before the
readback runs on GPU or CPU. Requires *many repeated* readbacks (a full-frame
1080p→2160p render loop, ~48+ frames); a single call always works. ~50 %
repro rate per run in a 48-frame loop on our hardware; observed to be fully
absent with autotune disabled.

### Reproducer

`crates/senmei-pipeline/tests/bench.rs::bench_upscale_step` (the `Upscale`
step) with `BENCH_MODEL=fallin-soft`:
`cargo test -p senmei-pipeline --release --test bench bench_upscale_step -- --ignored --nocapture`

The tiled-infer bench (`bench_upscaler_1080p_fullframe`) never panicked in the
timing loop. The **full-frame** fused step (`infer_rgb8`, whole 1080p in one
pass) fails — historically intermittently with the ordering panic, and since
2026-08-18 **deterministically with an OOM** (Bug 3) that then cascades into
the ordering/tuner panics. The two are symptoms of the same broken
cubecl-autotune async machinery under full-frame load. `infer_rgb8` is now
**tiled internally** (512px tiles) and is reliable — the fix.

### Root cause (analysis)

`OrderedExecution::execute_optimization` compares an optimization's `ordering`
(an `Arc<Vec<usize>>` of operation indices captured at **plan** time) against
`self.operations` (the queue at **execution** time):

```rust
if ordering.len() > self.operations.len() {
    panic!("Ordering is bigger than operations");
}
```

When the queue is drained between plan and execute (e.g. a readback-triggered
`drain_stream` on one stream while cubecl-autotune is asynchronously
benchmarking/tuning a fused kernel on another DSU worker), the optimization
runs with a stale ordering that references more operations than remain →
panic. Mixing small tile readbacks and large full-frame readbacks increases the
queue-drain/plan interleavings that hit this window, but a pure full-frame
readback loop also trips it (see Bug 3 — the panics can also be a cascade from
a corrupted server). This is a thread-safety bug in the fusion queue
bookkeeping; the concurrency that triggers it comes from cubecl-autotune's
async tuning work.

### Workarounds tried (results)

| Workaround | Result |
|---|---|
| Read back f32 instead of f16/u8 (`into_data().convert::<f32>().to_vec()`) | Helps in isolation; does not eliminate the failures under load. |
| Move permute/clamp/scale to CPU (plain forward + f32 readback) | Does not eliminate the failures. |
| Pure `infer_rgb8` only (no tiled prefix) | Still fails (see Bug 3) — deterministic OOM on a huge autotune matmul. |
| **`infer_rgb8` tiled internally (512px)** | **Reliable** (48-frame loop, correctness vs reference within fp16) — the shipped fix. |
| Autotune OFF (feature flag off, fusion ON) | Reliable but ~5× slower (1025 ms vs ~195 ms/frame). |
| Fusion OFF (autotune ON) | Panics in the cubecl tuner (Bug 2). |

### Upstream status

Still present on `main` (checked 2026-08-18): the panic was kept, only the
message got more verbose (adds `ordering len`, `operations len`,
`num_executed`, `optimization len`). No fix upstream.

### Impact on Senmei

Fixed: `infer_rgb8` now tiles internally (512px, overlap-stitched u8) so no
full-frame matmul reaches autotune — structurally immune to the OOM. The
`senmei-pipeline` bench isolates the two measurements
(`bench_upscaler_1080p_fullframe` = tiled infer, `bench_upscale_step` = fused
step, fresh engine each). Cost: tiling re-computes ~2× pixels (overlap), so
the step is ~329 ms / 3.0 FPS (fallin-soft) vs 227 ms for the full-frame
CPU-convert path — overlap / GPU-stitch tuning is tracked in docs/todos.md.

### Suggested upstream issue

**Title:** `burn-fusion: intermittent "Ordering is bigger than operations" panic with autotune enabled (stale ordering vs drained queue)`

**Body sketch:**
1. Env: burn 0.21.0 + burn-wgpu (Vulkan), cubecl 0.10.0, AMD RADV/RDNA4.
2. Fused output readback panics after many frames in `ordering.rs` (see above).
   ~50 % per run; single call always fine; small tile readbacks rarely hit it,
   large full-frame readbacks do.
3. Analysis: optimization `ordering` (plan-time snapshot) vs `operations`
   (execution-time queue) can diverge when the queue is drained while
   autotune tunes asynchronously. `execute_optimization` asserts instead of
   re-validating/re-planning.
4. Proposal: instead of panicking when `ordering.len() > operations.len()`,
   re-plan or fall back to executing the available operations unfused (a
   graceful skip keeps correctness — the ops stay queued — and loses only the
   fusion for that batch).

---

## Bug 2 — cubecl: tuner panic when autotune runs without fusion

### Symptom

With `burn-wgpu` autotune **enabled** but fusion **disabled**
(`default-features = false, features = ["std", "vulkan"]`), inference panics in
the cubecl tuner while executing the autotune plan:
`cubecl-runtime/.../tune/tuner.rs` (no fused path to satisfy the plan).

### Status

Secondary manifestation of the same autotune machinery; we did not pursue it
(autotune+fusion is our shipped config). Worth mentioning if filing about the
tuner's expectation that fusion is present.

---

## Bug 3 — cubecl-wgpu: OOM while autotune benchmarks a huge full-frame matmul

### Symptom

With autotune ON, running the full-frame (non-tiled) fused `infer_rgb8` path
repeatedly now fails **deterministically** (~4.6 s in, 4/4 runs, 2026-08-18):

```
cubecl-wgpu .../compute/server.rs:270: can't allocate buffer of size: 4395368448
```

This is during autotune benchmarking of a large matmul
`MatmulAutotuneKey { m: 1024, n: 4194304, k: 64, f16 }` that only appears in
the **full-frame** forward (the tiled `infer` path never produces a 4M-column
matmul). The failed 4.4 GB allocation leaves the cubecl server in an invalid
state ("Memory page 0 doesn't exist"), which then surfaces as the tuner panic
(Bug 2) and/or the ordering panic (Bug 1) — so the OOM is likely the root
trigger behind many of the "intermittent" ordering panics too.

### Status / impact

No upstream fix. This is what makes the app's render path unreliable: when the
autotune cache misses a full-frame shape, tuning OOMs, the server corrupts and
the render aborts (output file deleted). The tiled path avoids the 4M matmul
but is not what `Upscale::infer_rgb8` uses. Reliable only with autotune OFF.
Suggested issue: cubecl-wgpu autotune must not OOM the device when benchmarking
a large shape — either skip oversized candidates or recover from a failed
allocation instead of corrupting the server.

---

## Repo-side action items

- [ ] File Bug 1 + Bug 3 upstream (tracel-ai/burn / tracel-ai/cubecl), link here.
- [ ] Decide the app default: keep autotune ON (fast, occasional failure) vs
      OFF (reliable, 5× slower) vs vendor-patch with graceful recovery. See
      `docs/todos.md`.
