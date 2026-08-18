# burn (tracel-ai/burn) upstream bugs — findings

Bugs found running `burn-wgpu` 0.21 / `cubecl` 0.10 on AMD RADV (RX 9070 XT,
RDNA4, `gfx1201`, non-conformant Vulkan), kept here so each finding can be
turned into a proper upstream issue. Versions: `burn = 0.21.0`, `burn-wgpu =
0.21.0` (features `vulkan`), `burn-fusion = 0.21.0`, `cubecl = 0.10.0`.

---

## Bug 1 — burn-fusion: intermittent "Ordering is bigger than operations" panic

**Symptom.** Deterministic eventually, nondeterministic when: a fused tensor
readback panics on the fusion server's worker thread (`DSU-0-0`) at
`burn-fusion/src/stream/execution/ordering.rs:49`:

```
thread 'DSU-0-0' panicked at .../burn-fusion-0.21.0/src/stream/execution/ordering.rs:49:13:
Ordering is bigger than operations
```

The caller then unwraps the `CallError` at `client.rs:175/189`
(`read_tensor_int`/`read_tensor_float`) and panics too. Happens with **any**
model (Real-CUGAN, Fallin/UpCunet2xFast), **any** readback dtype (f16, u8,
f32), and regardless of whether the conversion runs on GPU or CPU. Requires
*many* readbacks (a full-frame 1080p→2160p render loop, 48+ frames); a single
call always works. ~50 % repro rate per 48-frame run; absent with autotune
disabled.

**Reproducer.**
`crates/senmei-pipeline/tests/bench.rs::bench_upscale_step` (the `Upscale`
step) with `BENCH_MODEL=fallin-soft`:
`cargo test -p senmei-pipeline --release --test bench bench_upscale_step -- --ignored --nocapture`

The tiled-infer bench (`bench_upscaler_1080p_fullframe`) never panicked in the
timing loop. The **full-frame** fused step (`infer_rgb8`, whole 1080p in one
pass) fails — historically intermittently with the ordering panic, and since
2026-08-18 **deterministically with an OOM** (Bug 3) that then cascades into
the ordering/tuner panics. Both are symptoms of the same broken cubecl-autotune
async machinery under full-frame load. `infer_rgb8` is now **tiled internally**
(640px tiles) and reliable — the fix.

**Root cause.** `OrderedExecution::execute_optimization` compares an
optimization's `ordering` (an `Arc<Vec<usize>>` of operation indices captured
at **plan** time) against `self.operations` (the queue at **execution** time):

```rust
if ordering.len() > self.operations.len() {
    panic!("Ordering is bigger than operations");
}
```

If the queue is drained between plan and execute (e.g. a readback-triggered
`drain_stream` on one stream while cubecl-autotune asynchronously benchmarks a
fused kernel on another DSU worker), the stale ordering references more ops
than remain → panic. Mixing small tile readbacks and large full-frame readbacks
increases the queue-drain/plan interleavings that hit this window. A
thread-safety bug in the fusion queue bookkeeping; the concurrency comes from
cubecl-autotune's async tuning.

**Workarounds tried.**

| Workaround | Result |
|---|---|
| Read back f32 instead of f16/u8 (`into_data().convert::<f32>().to_vec()`) | Helps in isolation; does not eliminate under load. |
| Move permute/clamp/scale to CPU (plain forward + f32 readback) | Does not eliminate. |
| Pure `infer_rgb8` only (no tiled prefix) | Still fails (Bug 3) — deterministic OOM on a huge autotune matmul. |
| **`infer_rgb8` tiled internally (640px)** | **Reliable** (48-frame loop, correctness vs reference within fp16) — the shipped fix. |
| Autotune OFF (feature flag off, fusion ON) | Reliable but ~5× slower (1025 ms vs ~195 ms/frame). |
| Fusion OFF (autotune ON) | Panics in the cubecl tuner (Bug 2). |

**Upstream status.** Still present on `main` (checked 2026-08-18): the panic was
kept, only the message got more verbose (adds `ordering len`, `operations
len`, `num_executed`, `optimization len`). No fix upstream.

**Impact on Senmei.** Fixed: `infer_rgb8` now tiles internally (640px) so no
full-frame matmul reaches autotune — structurally immune to the OOM. Tiles are
accumulated into one f16 canvas on the GPU (`slice_assign` overlap averaging)
and read back as a single packed frame, so the earlier per-tile u8 readback +
CPU stitch cost is gone: `bench_upscale_step` (fallin-soft) 329 → **186.1 ms /
5.4 FPS**. The `senmei-pipeline` bench isolates the two measurements
(`bench_upscaler_1080p_fullframe` = tiled infer, `bench_upscale_step` = fused
step, fresh engine each). Tiling still re-computes ~2× pixels (overlap) — the
remaining gap to the full-frame CPU-convert path (227 ms) is overlap re-compute,
not readback.

**Suggested upstream issue.** Title: `burn-fusion: intermittent "Ordering is
bigger than operations" panic with autotune enabled (stale ordering vs drained
queue)`.
1. Env: burn 0.21.0 + burn-wgpu (Vulkan), cubecl 0.10.0, AMD RADV/RDNA4.
2. Fused output readback panics after many frames in `ordering.rs` (see above).
   ~50 % per run; single call always fine; small tile readbacks rarely hit it,
   large full-frame readbacks do.
3. Analysis: optimization `ordering` (plan-time snapshot) vs `operations`
   (execution-time queue) can diverge when the queue is drained while autotune
   tunes asynchronously. `execute_optimization` asserts instead of
   re-validating/re-planning.
4. Proposal: instead of panicking when `ordering.len() > operations.len()`,
   re-plan or fall back to executing the available operations unfused (a
   graceful skip keeps correctness — the ops stay queued — and loses only the
   fusion for that batch).

---

## Bug 2 — cubecl: tuner panic when autotune runs without fusion

With autotune **enabled** but fusion **disabled** (`default-features = false,
features = ["std", "vulkan"]`), inference panics in the cubecl tuner while
executing the autotune plan (`cubecl-runtime/.../tune/tuner.rs`) — no fused
path satisfies the plan. Secondary manifestation of the same autotune
machinery; not pursued (autotune+fusion is the shipped config). Worth
mentioning when filing about the tuner's expectation that fusion is present.

---

## Bug 3 — cubecl-wgpu: OOM while autotune benchmarks a huge full-frame matmul

**Symptom.** With autotune ON, running the full-frame (non-tiled) fused
`infer_rgb8` path repeatedly now fails **deterministically** (~4.6 s in, 4/4
runs, 2026-08-18):

```
cubecl-wgpu .../compute/server.rs:270: can't allocate buffer of size: 4395368448
```

This is during autotune benchmarking of a large matmul `MatmulAutotuneKey { m:
1024, n: 4194304, k: 64, f16 }` that only appears in the **full-frame** forward
(the tiled `infer` path never produces a 4M-column matmul). The failed 4.4 GB
allocation leaves the cubecl server in an invalid state ("Memory page 0 doesn't
exist"), which then surfaces as the tuner panic (Bug 2) and/or the ordering
panic (Bug 1) — so the OOM is likely the root trigger behind many of the
"intermittent" ordering panics.

**Status / impact.** No upstream fix. When the autotune cache misses a
full-frame shape, tuning OOMs, the server corrupts and the render aborts
(output file deleted). The tiled path avoids the 4M matmul but is not what
`Upscale::infer_rgb8` uses. Reliable only with autotune OFF. Suggested issue:
cubecl-wgpu autotune must not OOM the device when benchmarking a large shape —
either skip oversized candidates or recover from a failed allocation instead of
corrupting the server.

---

## Bug 4 — burn-nn GroupNorm breaks on f16 when the per-group element count is large

**Symptom.** `GroupNorm` on `Vulkan<f16>` divides the channel-sum by the
per-group element count via `div_scalar`. For per-group counts ≥ 2¹⁴ (e.g.
RealPLKSR's `GroupNorm(4, 64)` on 64×64 maps → 65536/group), the f16 reciprocal
`1/N` underflows to a subnormal that the fused kernel flushes to 0, so
`mean`/`var` collapse to 0 and the normalized output explodes (observed `±318`
vs torch `±1.3`). `mean_dim` (native scaled kernel) stays accurate, so the
workaround is to compute the norm with `mean_dim` instead of `sum + div_scalar`
(`crates/senmei-ml/src/burn/real_plksr.rs::group_norm`).

**Reproducer.** `crates/senmei-ml/src/burn/real_plksr.rs` tests: `sum_dim(2)` of
65536 × 0.5 is correct (32768), but `sum_dim(2).div_scalar(65536.0)` returns
0.0 while `mean_dim(2)` returns 0.5. Root cause is the reciprocal `1/65536`
being a subnormal f16 (flushed to zero); the sum itself only overflows for
≥ 65505.

---

## Bug 5 — burn-store `PytorchReader` ignores tensor strides (non-contiguous .pth)

**Symptom.** The pickle reader parses `args[3]` (stride) in
`rebuild_tensor_v2` but discards it (`// args[3] is stride (unused)`), reading
every tensor's storage linearly. A `.pth` whose weights were saved
non-contiguous (e.g. `4x_Alchemy` stores conv weights channels-last: shape
`[o,i,k,k]` but strides `(27,1,9,3)`) therefore loads **scrambled** weights
silently — the model runs but produces garbage (verified: head weight
correlation 0.24 vs the correct tensor).

**Workaround.** Preprocess the state dict with `torch` before converting:
`{k: v.contiguous() for k, v in sd.items()}` (the `deh264`/`dejpg` RealPLKSR
pths are already contiguous; only `4x_Alchemy` is channels-last). Suggested
upstream fix: capture `args[3]` and scatter the storage into the logical layout
in `rebuild_tensor_impl`.

---

## Bug 6 — burn-fusion: split/cat side-channel graph computes wrong results when compiled as a standalone function stream

**Symptom.** The IFRNet `ResBlock` ("side channels": `out[:,-side:] ← conv2(...)`
via channel `split`/`slice` + `cat`, interleaved with full-width convs) gives a
wrong output (mae 0.0525 vs torch on the decoder-4 ResBlock) whenever the
sequence runs through a **function call** (`ResBlock::forward`, or an inline
closure). The byte-identical op sequence executed **inline** in the test's main
flow is correct (mae ~0.0001). Deterministic across runs and processes; affects
every channel-slicing variant tried: `slice_assign`, `split_with_sizes`+`cat`,
`slice`+`cat`, `slice(...)*1.0`, and a mask-multiply rewrite with padded
full-width convs + `burn::tensor::module::conv2d` (mask version mae 0.098).

**Reproducer.** `crates/senmei-ml/src/burn/ifrnet.rs` test
`ifrnet_resblock_isolated` (removed after diagnosis): fresh `IfrNet<Vulkan<f16>>`
loaded from `IFRNet_Vimeo90K.pth.f16.bpk`; feeding `d4_cb0.bin` through
`decoder4.cb1.forward(...)` gives mae 0.0525 vs the torch `rb_s5.bin` reference,
while the same steps written inline give 0.00003. Encoder (plain conv+prelu
chains) is correct through methods, so the trigger is the channel-split/`cat`
graph, not function calls in general.

**Root cause (suspected).** The fused kernel compiled for the standalone
ResBlock stream reads the channel-sliced view with a wrong base offset / stride
(or an autotune cache collision for that fused graph); inline, the ops fuse into
a different stream with different surrounding tensors and stay correct. Not
pursued further — matches the repo's other broken-autotune/fusion findings
(Bugs 1–3) on RADV.

**Workaround / status.** None found that keeps the fused `Vulkan<f16>` backend
correct. IFRNet stays `loadable: false` ("arch port verified vs torch when
unfused; blocked by burn-fusion Bug 6"). Options: (a) file upstream with the
minimal repro, (b) run IFRNet on a non-fused backend (not viable repo-wide:
Bug 2), (c) revisit after a burn/cubecl update.

---

## Repo-side action items

- [x] Filed upstream (2026-08-18, author zachelnet):
  - Bug 1 → root-cause comment on `tracel-ai/burn#4950`
  - Bug 3 → `tracel-ai/cubecl#1531` (wgpu/autotune-OOM; related: `#1384` closed, `#1401` CUDA)
  - Bug 4 → `tracel-ai/burn#5382` (GroupNorm f16 div_scalar)
  - Bug 5 → `tracel-ai/burn#5383` (PytorchReader strides)
- [ ] Decide the app default: keep autotune ON (fast, occasional failure) vs
      OFF (reliable, 5× slower) vs vendor-patch with graceful recovery. See
      `docs/todos.md`.
