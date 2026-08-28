# Upstream reports

Consolidated tracking for upstream burn/cubecl/onnx findings — the former
`docs/burn-bugs.md` was merged here (2026-08-19). Each section is one upstream
issue: the **Finding** (symptom / root cause / workaround / status) plus the
**paste-ready text** to file or post upstream. Copy the text block as-is into
the linked issue or comment form.

Filed 2026-08-18 (author zachelnet): Bug 1 → comment on `tracel-ai/burn#4950`,
Bug 3 → `tracel-ai/cubecl#1531`, Bug 4 → `tracel-ai/burn#5382`, Bug 5 →
`tracel-ai/burn#5383`. Filed 2026-08-20: §6 → `tracel-ai/cubek#519`.
Section numbers ≠ old bug numbers; Bug 2 (tuner panic
without fusion) is a secondary manifestation of the same autotune machinery and
is folded into §2; Bug 7 (NAFNet fp16 overflow) is inherent to the weights +
input, not an upstream bug — its finding is in `docs/models.md` (Notes).

**Status 2026-08-25**: §3 (GroupNorm), §5 (ONNX reader) and §7 (burn-tch
dlopen, not planned) are **closed** upstream; §1, §2, §4, §6 remain **open**.
None of the fixes are in our pinned burn fork (`v0.21.0-senmei-burn-store-strides`),
so all local workarounds stay in place until a fork bump pulls them in.

---

## 1. burn-fusion "Ordering is bigger than operations" — Bug 1 (burn#4950)

**Finding.** A fused tensor readback panics on the fusion server's worker thread
at `burn-fusion/src/stream/execution/ordering.rs:49` ("Ordering is bigger than
operations") after many readbacks under autotune on Vulkan (AMD RADV/RDNA4);
single calls never panic. Root cause: `execute_optimization` compares the
optimization's plan-time `ordering` against the execution-time `operations`
queue; a drain between plan and execute leaves a stale ordering → panic.
Workaround: `infer_rgb8` is tiled internally (640 px) so no full-frame matmul
reaches autotune — reliable, 186 ms / 5.4 FPS; autotune OFF is reliable but ~5×
slower. Upstream status: **fixed on burn `main`** — maintainer answer
(2026-08-21, laggui): #4962 (cross-thread handle lifetime), #5282 (readbacks no
longer cut pending fusion compositions), #5400 (cached plan no longer applied to
a shorter segment = the `ordering.len() > operations.len()` condition directly).
Requires the 0.22 API migration (backend generics removed from the user API).
`main` now also prints ordering/op-count/optimization lengths to distinguish a
stale plan from a poisoned queue.
**Probe 2026-08-21**: the panic does **not** reproduce in isolation on 0.21.0 —
300–400 fused matmul chains + sync `into_data()` readbacks with cycling
autotune keys (re-benchmark every iter) run clean on both 0.21.0 and `main`.
Consistent with laggui's "often secondary to an earlier missing-handle or
kernel failure": in our flow it fired after the cubecl#1531 OOM corrupted the
server; on cubecl `main` that precursor no longer corrupts the server (§2).
**Status 2026-08-25**: issue still **open** (burn#4950). nathanielsimard: no
fallback execution mode planned (protecting input handles would kill in-place
ops); kernel-selection work (tile API) should make it moot. laggui: #4962/#5282/#5400
landed; `main` now prints ordering/op/optimization lengths to distinguish stale
plan vs poisoned queue. Our last comment: will re-validate after the 0.22
migration (burn removes backend generics from the user API).

**Paste-ready text** — post as a comment on burn#4950:

```text
We hit the same panic deterministically under autotune on Vulkan
(burn 0.21.0 + burn-wgpu Vulkan, cubecl 0.10.0, AMD RADV/RDNA4). Reproducer: a
fused output readback (Tensor::into_data().to_vec()) repeated over ~48 frames of
a 1080p→2160p super-resolution render panics ~50% of the time; a single call
never panics, and small tile readbacks rarely do.

Root cause: OrderedExecution::execute_optimization compares an optimization's
`ordering` (an Arc<Vec<usize>> of operation indices captured at plan time)
against `self.operations` (the queue at execution time):

    if ordering.len() > self.operations.len() {
        panic!("Ordering is bigger than operations");
    }

When the queue is drained between plan and execute (a readback triggers
drain_stream on one stream while cubecl-autotune asynchronously benchmarks a
fused kernel on another DSU worker), the stale ordering references more ops than
remain → panic. The interleaving is a queue-bookkeeping bug made reachable by
autotune's async tuning; mixing small tile readbacks and large full-frame
readbacks changes how often the window is hit.

Proposal: instead of panicking when ordering.len() > operations.len(), re-plan or
fall back to executing the available operations unfused. That keeps correctness
(the ops stay queued) and loses only fusion for that batch.
```

---

## 2. cubecl-wgpu: autotune OOMs the device on an oversized matmul, then corrupts the server — Bug 3 (cubecl#1531)

**Finding.** With autotune ON, a full-frame fused `infer_rgb8` fails
deterministically (~4.6 s in, 4/4 runs, 2026-08-18):
`cubecl-wgpu .../compute/server.rs:270: can't allocate buffer of size: 4395368448`
— autotune benchmarks `MatmulAutotuneKey { m: 1024, n: 4194304, k: 64, f16 }`, a
shape only a full-frame forward produces. The failed 4.4 GB allocation leaves
**Status 2026-08-25**: issue still **open** (cubecl#1531). nathanielsimard asked
if it still reproduces on latest burn/cubecl/cubek; we confirmed: on pinned
cubecl 0.10.0 the repro stands (24 GiB reserve on 16 GiB card corrupts the
server), on `main` (post-#1494) the failed reserve still panics the wgpu
worker but the server **no longer stays corrupted** — persistent-corruption
symptom fixed, allocation not yet graceful. #1494's lazy handle→memory
binding should stop autotune dry-run materializing the 4.4 GB candidate;
committed to re-test once v0.11.0 is out.

the server invalid ("Memory page 0 doesn't exist"), surfacing as the tuner panic
(Bug 2) and/or the ordering panic (Bug 1). Workaround: the tiled path (640 px)
avoids the 4M-column matmul. Related: `#1384` closed, `#1401` CUDA.
Upstream: #1494 (memory pools + **lazy handle→memory binding**, so autotune
dry-run no longer materializes oversized candidates) merged on `main`
2026-08-12, not yet in a release; #1438 (retry allocation after reclaiming
cached pages) closed unmerged. **Verified on cubecl `main` 2026-08-21**
(standalone Vulkan probe, 24 GiB reserve on 16 GiB card): a failed allocation
still panics the wgpu server worker (`server.rs:318 failed to reserve`) and that
launch's `sync()` errors "server is in an invalid state", but the server **no
longer stays corrupted** — subsequent operations succeed (readback correct).
So the persistent corruption symptom is fixed; the failure is recoverable but
not yet graceful.

**Paste-ready text** (Title + Body):

```text
cubecl-wgpu: autotune OOMs the device benchmarking an oversized matmul, then corrupts the server
```

Body:

```text
**Describe the bug**
With autotune enabled, benchmarking a large matmul during tuning can attempt to
allocate a buffer larger than the device, and instead of failing gracefully it
corrupts the compute server so every subsequent operation fails with "Memory
page 0 doesn't exist" (the stale-page symptom from #1384 / #1401, but a
different trigger — this one is an autotune-caused OOM, not the renumbering bug).

    cubecl-wgpu .../compute/server.rs:270: can't allocate buffer of size: 4395368448

The failing key is `MatmulAutotuneKey { m: 1024, n: 4194304, k: 64, f16 }`
(a 4.4 GB tuning candidate) — only produced by a full-frame super-resolution
forward (1080p→2160p). The failed allocation leaves the server invalid, which
then surfaces as the tuner panic and/or the burn-fusion "Ordering is bigger than
operations" panic.

**Environment**
- cubecl 0.10.0 + cubecl-wgpu (Vulkan), burn 0.21.0 + burn-wgpu
- AMD RX 9070 XT (RDNA4), RADV (non-conformant), 16 GB VRAM, Fedora Linux

**Expected**
Autotune should either skip candidates that don't fit, or recover from a failed
allocation (drop the candidate) instead of corrupting the server.

**Workaround**
Tile the input so no full-frame matmul reaches autotune (we run 640px tiles).
```

---

## 3. burn-nn GroupNorm: f16 div_scalar underflows for large per-group element counts — Bug 4 (burn#5382)

**Finding.** `GroupNorm` on `Vulkan<f16>` divides the channel-sum by the
per-group element count via `div_scalar`; for counts ≥ 2¹⁴ the f16 reciprocal
1/N is subnormal and the fused kernel flushes it to 0 → normalized output
explodes (observed ±318 vs torch ±1.3). `mean_dim` (native scaled kernel) stays
accurate. Workaround: compute the norm with `mean_dim` instead of
`sum + div_scalar` (`crates/senmei-ml/src/burn/real_plksr.rs::group_norm`).
Upstream: #5211 fixed the **norm accumulation** part, and **#5410
("fix(nn): use mean reduction in group norm", laggui, merged 2026-08-21,
commit 8832b00)** fixed the **denominator division** the same way we do —
`sum_dim(2) / N` → `mean_dim(2)`, with a test that reproduces our exact case
(GroupNorm(4, 64), 65536/group, f16). Issue closed as completed. Our `mean_dim`
workaround is byte-for-byte the upstream approach, so it stays until a burn
bump pulls in the fix, then the custom `group_norm` helper in
`real_plksr.rs` can be deleted in favour of burn's native GroupNorm.
**Status 2026-08-25**: **closed as completed** via `tracel-ai/burn#5410`
("fix(nn): use mean reduction in group norm", laggui, merged). Fix is on burn
`main`, **not yet in our pinned fork** (`v0.21.0-senmei-burn-store-strides`),
so the `mean_dim` workaround in `real_plksr.rs` stays until a fork bump pulls
it in.

**Paste-ready text** (Title + Body):

```text
burn-nn GroupNorm: f16 div_scalar underflows for large per-group element counts
```

Body:

```text
**Describe the bug**
GroupNorm on Vulkan<f16> computes the mean by summing and dividing by the
per-group element count via div_scalar. For per-group counts ≥ 2^14 the f16
reciprocal 1/N is subnormal and is flushed to 0 by the fused kernel, so the
normalized output explodes.

**To Reproduce**
1. Create an f16 tensor of 65536 elements of 0.5 (or run RealPLKSR's
   GroupNorm(4, 64) on a 64×64 map → 65536 elements per group).
2. sum_dim(2).div_scalar(65536.0) returns 0.0 (should be 0.5).
3. mean_dim(2) returns 0.5 (correct).

Real case: the normalized output was ±318 vs the torch reference ±1.3.

**Expected behavior**
div_scalar should not underflow on f16 — either promote the reciprocal to f32,
or compute the mean with a scaled kernel like mean_dim.

**Desktop (please complete the following information):**
 - OS: Fedora Linux
 - burn: 0.21.0 (burn-nn), burn-wgpu (Vulkan)
 - GPU: AMD RX 9070 XT (RDNA4), RADV, 16 GB VRAM

**Additional context**
Workaround: compute the normalization with mean_dim instead of sum + div_scalar.
```

---

## 4. burn-store PytorchReader ignores tensor strides — Bug 5 (burn#5383)

**Finding.** The pickle reader parses `args[3]` (stride) in `rebuild_tensor_v2`
but discards it, reading storage linearly — a non-contiguous `.pth`
(channels-last, e.g. `4x_Alchemy`: shape `[o,i,k,k]`, strides `(27,1,9,3)`)
loads silently scrambled (head weight correlation 0.24). Workaround: preprocess
the state dict with `{k: v.contiguous() for k, v in sd.items()}`. Confirmed
again on real production weights (2026-08-21): TNTwise/Phhofm `params`-wrapped
SPAN checkpoints (`2xHFA2kSPAN`, `2x_ModernSpanimationV1/V1.5`,
`2xBHI_small_span_pretrain`) store their 3×3 `conv1` weights non-contiguous
(shape `[128,6,3,3]`, strides `(54,1,18,6)`), so every 3×3 kernel loads in
`(out,kh,kw,in)` order and the models render inverted/scrambled; 1×1 kernels
(conv0/conv2/sk) stay correct since kh=kw=1 moves nothing. The contiguous
preprocess fixes all of them (verified end-to-end).

**Fix PR** (2026-08-20): `tracel-ai/burn#5392` „fix(store): respect PyTorch
tensor strides" — parses/validates strides in both rebuild paths and materializes
non-contiguous views in logical row-major order. Still open (needs review;
patch coverage 69% < 80% target). Once merged + released, the contiguous
preprocess step can be dropped from the SPAN/RealPLKSR/SCUNet convert flow.
**Status 2026-08-25**: issue still **open** (burn#5383); PR #5392 (by
original4422) still **open/unmerged**. Workaround (`{k: v.contiguous()}` in the
convert flow) stays until it lands.

**Paste-ready text** (Title + Body):

```text
burn-store PytorchReader ignores tensor strides (non-contiguous .pth loads scrambled)
```

Body:

```text
**Describe the bug**
PytorchReader parses the stride tuple (args[3] in rebuild_tensor_v2) but
discards it, reading every tensor's storage linearly. A .pth whose weights were
saved non-contiguous therefore loads silently scrambled — the model runs but
produces garbage.

**To Reproduce**
1. Save a conv weight tensor non-contiguous (channels-last: shape [o,i,k,k],
   strides (27,1,9,3)) into a .pth.
2. Load it with PytorchStore.
3. Compare against the original: the loaded head weights have correlation 0.24
   (scrambled).

**Expected behavior**
Capture the stride and scatter the storage into the logical layout (or
.contiguous() the data before returning), matching torch's semantics.

**Desktop (please complete the following information):**
 - OS: Fedora Linux
 - burn: 0.21.0 (burn-store)

**Additional context**
Workaround: pre-process the state dict with
{k: v.contiguous() for k, v in sd.items()}.
```

---

## 5. burn-onnx: runtime API to load ONNX initializer tensors — feature request (burn-onnx#456)

**Finding.** Feature request, not a bug: projects using ONNX only as a weight
container (hand-ported architecture) need a runtime initializer reader without
the codegen step. No duplicate found — existing issues cover operator coverage /
codegen, not a weight-only initializer reader.
**Status 2026-08-25**: issue **closed as completed** (burn-onnx#456). antimora:
the public `onnx-ir` API already covers this on 0.21.0 — `ModelProto`/
`TensorProto` are re-exported and `impl TryFrom<TensorProto> for TensorData`
works; no codegen, no graph building. Three gotchas: (1) also read `Constant`
nodes (tensor in `value` attribute), not just `graph.initializer` — 92/570
vendored models have tensor data only in constants; (2) key constants by the
node's **output** name (inner `TensorProto.name` is "value"/empty); (3) external
data (`data_location == EXTERNAL`) is unreachable from a `&[u8]` API. He closed
the feature request (the API can't say which tensors are weights — that's an
arch-specific heuristic). Our dependency-free reader already implements all
three gotchas (`c2d1703`), so we keep it; `ParseError` re-export fix upstream
noted.

**Paste-ready text** (Title + Body):

```text
feat: runtime API to load ONNX initializer tensors (weight-container use case)
```

Body:

```text
<!-- Please search existing issues to avoid creating duplicates -->

### Feature description

A runtime-only API that reads just the `initializer` tensors from an ONNX file
as plain tensors, without the codegen step. Today burn-onnx converts the full
graph into generated Rust code; projects that only use ONNX as a weight
container (hand-ported architecture) need the weights, not the graph.

### Feature motivation

We (an open-source video enhancer, MIT OR Apache-2.0, on burn 0.21.0) hand-port
architectures as native burn code and only need the weights. Several checkpoints
ship exclusively as ONNX, so we maintain a dependency-free hand-rolled protobuf
reader to extract the `initializer` tensors. That duplicates serialization logic
already implemented in burn-onnx. A public API would let us delete the parser
and stop duplicating ONNX wire-format handling.

### (Optional) Suggest a Solution

Surface the existing initializer-parsing path as a public runtime function, e.g.
in burn-onnx:

    pub fn load_initializers(bytes: &[u8]) -> Result<Vec<(String, TensorData)>>;

The protobuf tensor wire format is already parsed (and covered by burn-onnx
tests); the change is mostly exposing it publicly and decoupling it from graph
codegen. Tradeoff: keep it runtime-only (no generated code) so it stays a small,
dependency-light path for weight loading.
```

## 6. cubek-convolution f16 1×1 conv wrong for K=96 × N≥32768 — Bug (cubek)

**Finding.** Root cause of the SPAN f16 degradation is now isolated (2026-08-20):
it is **not** accumulation precision, **not** silu/sigmoid, and **not** weight
quantization. A bit-exact op-by-op diff (burn f16 vs torch ROCm f16) showed the
norm output and **every conv weight/bias are bit-identical**, `conv0` (3→6 1×1)
and `sk` (3→48 1×1) differ by ~1 ULP, but `conv2` (96→48 1×1, the last conv of
the first SPAB `Conv3XC`) produces **wrong values** (mean abs err 1.1, max 15.3),
not rounded ones. Minimal standalone repro confirms it is a shape-dependent
kernel bug in the f16 implicit-GEMM conv, independent of data:

- `Conv2d([96, 48], [1, 1])` f16 on Vulkan, input `[1, 96, H, W]`:
  - `H·W ≤ 16384` → correct (≤ 1–2 ULP)
  - `H·W ≥ 32768` → **wrong** (mean abs err ≈ 1.0, max ≈ 7.7; e.g. 128×256,
    240×320, 256×256)
- Same `H·W = 76800` but `K ∈ {48, 64, 80, 97, 112, 128}` → correct.

So the trigger is the specific combination **K=96 (in channels) AND N ≥ 32768
(spatial positions)**, not a precision choice (the matmul `Acc` is already
`(f16, f32)` on Linux). In SPAN this hits every `conv2` (96→48) at full frame
240×320 = 76800, and the error compounds ~2× per block. Workaround: run SPAN in
f32, or use only 64ch / no_norm checkpoints (their 1×1 convs never hit K=96 at
this N).

**Root cause (2026-08-28):** Virtual-vs-physical padding mismatch in the
cubek-convolution async-copy path. Not a kernel logic bug, but a missing memory
barrier between logical padding and physical allocation:

1. `adjust_problem` (`kernels/forward/args.rs:51-67`) computes
   `padded_channels = 96.next_multiple_of(channel_align) = 128` and sets
   `problem.k = 128`. The `Im2colLayout` and `WeightLayout` use this padded
   K for tiling.
2. `into_contiguous_pitched` (`routines/base.rs:44-55`) converts NCHW→NHWC
   but **does not zero-pad the buffer** — the weight tensor stays
   `[48, 1, 1, 96]` in memory, not `[48, 1, 1, 128]`.
3. `WeightLayout::is_in_bounds` (`components/global/layout/weight.rs:102-107`)
   checks `k < self.rows` where `rows = problem.k = 128` (padded), not
   `shape_channel = 96` (actual tensor dim). The `Chain<NhwcLayout,
   WeightLayout>` composition ANDs both bounds checks, but the
   `async_copy_from` path (`cubek-matmul:global/read/strategy/async_copy.rs`)
   uses `view.shape()` which is set to the **padded** dimensions
   `(M, K) = (H·W, 128)`, so the copy never clips the read length.
4. Result: the async copy reads channels 96–127 from a 96-channel buffer →
   OOB garbage fed into the GEMM. The garbage values are non-zero (GPU
   allocator doesn't zero pages), corrupting the accumulation.

**Why only K=96 × N≥32768:**
- K=96 is the only value where `channel_align=64` produces `padded=128 = 2×K`.
  Other K values (80, 97, 112) also pad to 128, but the ratio `padded/K`
  differs; the tiling heuristic (`find_stage_size_m_n`) picks a different
  stage-K for those, splitting the K dimension across multiple load passes.
  At K=96 the full K fits in one stage-K pass, so every OOB position is
  guaranteed to be hit. At N≥32768 the M-dimension tiling produces enough
  work-groups that the async copy pipeline is saturated, preventing the
  sporadic page-zero fallback that sometimes masks the bug at smaller N.

**Fix paths (upstream):**
- **(A) Physical padding**: zero-pad input/weight to `padded_channels` in
  `correct_layout` before launching the kernel. Cleanest fix, ~1–2 days.
- **(B) Bounds-check fix**: make `WeightLayout::is_in_bounds` check against
  `shape_channel` (= actual tensor dim) instead of `self.rows` (= padded).
  ~1 day, but `async_copy_from` also needs updating to clip against the real
  shape.
- **(C) Checked async copy**: force `config.gmem_config.check_row_bounds =
  true` for the K dimension in the convolution path. ~0.5 days, may slow down
  all convos.

Repro artifacts kept in-repo: self-contained Rust test `conv1x1_repro` in
`crates/senmei-ml/src/burn/span.rs` (deterministic LCG data + f32 CPU
reference, no external files) and the torch cross-check `tools/span_conv_repro.py`.

Target repo: **`tracel-ai/cubek`** (crate `cubek-convolution`), not cubecl — the
conv kernel lives in cubek. Filed as **`tracel-ai/cubek#519`** (2026-08-20); no
duplicate at filing time (search `is:issue conv` returned only closed #20, a
bias-gradient reduce-sum bug).
**Workaround (2026-08-21)**: pad the 1×1 conv's input channels 96→128 (zero-pad
the weight into a K=128 conv + zero-pad the input at forward) — K=128 and K=192
are verified correct at N=76800, so `Span::pad_k96` unblocks the 48ch models.
Perf: K=128 padded is *not* slower than the broken K=96 (measured −9% on the
conv; K=128 tiles better). 4 disabled 48ch SPAN models re-enabled. The
forward-conv kernel (`kernels/forward/`) is unchanged since 2026-06-10, so the
upstream bug persists until a real fix.
**Status 2026-08-25**: issue still **open** (cubek#519); auto-added to the
CubeK project board as **Backlog**, no assignee, no linked PR. Workaround
(`Span::pad_k96`) stays.

**Paste-ready text** (Title + Body):

```text
cubek-convolution: f16 1x1 conv returns wrong results when in_channels=96 and spatial positions ≥ 32768
```

Body:

```text
<!-- Please search existing issues to avoid creating duplicates -->

### Describe the bug

On the wgpu backend (Vulkan, AMD RADV/RDNA4) a f16 1x1 convolution returns
numerically wrong results (not just rounded) for a specific shape: input
channels = 96 and H*W ≥ 32768. The same conv is bit-exact against a f32
reference for every other shape tested.

### To Reproduce

```rust
// burn 0.21 / burn-wgpu Vulkan, Backend = Vulkan<f16>
let mut conv = Conv2dConfig::new([96, 48], [1, 1]).init(&device);
// load any f16 weights + bias, e.g. random normal(0, 0.08)
let x = /* [1, 96, 128, 256] f16 input */;
let out = conv.forward(x); // wrong: mean abs err ~1.0 vs f32 reference
```

Shapes that reproduce (f16): `[1, 96, 128, 256]` (N=32768), `[1, 96, 240, 320]`
(N=76800), `[1, 96, 256, 256]` (N=65536).
Shapes that are correct: `[1, 96, 128, 128]` (N=16384), and `[1, K, 240, 320]`
for K in {48, 64, 80, 97, 112, 128}.

### Expected behavior

f16 conv with f32 accumulation should match the f32 reference to ~1 ULP,
independent of the shape.

### Environment

- burn / burn-wgpu / burn-cubecl 0.21.0, wgpu Vulkan on RADV
- GPU: AMD RX 9070 XT (gfx1201)
```

---

## 7. burn-tch: load libtorch at runtime (dlopen), no build-time link — feature request (burn#5416)

**Finding.** Feature request: `burn-tch` loads libtorch at runtime via dlopen
instead of inheriting `torch-sys`'s build-time link args. Motivation: a
distributable binary can compile without libtorch installed and pick the
CPU/CUDA/ROCm runtime at startup. We already maintain production forks for this
(`senmei-app/tch-rs` @ `v0.22.0-senmei-dlopen`, `senmei-app/burn` @
`v0.21.0-senmei-burn-tch-dlopen`) wired via `[patch.crates-io]`.
**Status 2026-08-25**: **closed as not planned** (burn#5416, laggui) — requires
runtime-loading support in `tch-rs`/`torch-sys` upstream; burn won't carry a
patched upstream dependency. Revisit if upstream merges/releases dlopen
support. Our forks remain the long-term path (already in production).

**Paste-ready text** (Title + Body):

```text
burn-tch: load libtorch at runtime (dlopen), no build-time link
```

Body:

```text
<!-- Please search existing issues to avoid creating duplicates -->

### Feature description

`burn-tch` loads libtorch at runtime (dlopen) instead of linking it at
build time.

Today `burn-tch` inherits the libtorch linking from `torch-sys` (via `tch`):
`torch-sys`'s build script emits libtorch link args, so libtorch must be
resolvable while the crate is being built. A `dlopen` mode would skip those
link args and resolve the `libtorch.so` symbols at runtime, so the crate
compiles without libtorch present and loads it on startup.

### Feature motivation

For distributable binaries (e.g. a desktop video app shipping on
Linux/Windows/macOS), build-time libtorch linking forces the build machine to
have libtorch installed or the binary to bundle it. Runtime loading lets the
same binary run against whatever libtorch the target system has, and lets the
user pick the runtime variant (CPU / CUDA / ROCm) at startup.

### (Optional) Suggest a Solution

A `dlopen` build feature in `burn-tch` (plus the matching `torch-sys`/`tch`
change) that:

- skips the build-time libtorch link args,
- resolves the libtorch symbols via `dlopen` at runtime,
- keeps the existing build-time-link behavior as the default (no breakage).

We already maintain working forks for this and use them in production:

- `senmei-app/tch-rs` @ `v0.22.0-senmei-dlopen` (dlopen in `torch-sys`/`tch`)
- `senmei-app/burn` @ `v0.21.0-senmei-burn-tch-dlopen` (burn-tch build script
  no longer adds libtorch link args)

wired via `[patch.crates-io]` for a Vulkan + optional ROCm/libtorch backend
(runtime variant selection). Happy to contribute this as a PR if it fits
`burn-tch`'s direction.
```


