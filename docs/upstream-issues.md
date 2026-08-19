# Upstream reports

Consolidated tracking for upstream burn/cubecl/onnx findings — the former
`docs/burn-bugs.md` was merged here (2026-08-19). Each section is one upstream
issue: the **Finding** (symptom / root cause / workaround / status) plus the
**paste-ready text** to file or post upstream. Copy the text block as-is into
the linked issue or comment form.

Filed 2026-08-18 (author zachelnet): Bug 1 → comment on `tracel-ai/burn#4950`,
Bug 3 → `tracel-ai/cubecl#1531`, Bug 4 → `tracel-ai/burn#5382`, Bug 5 →
`tracel-ai/burn#5383`. Section numbers ≠ old bug numbers; Bug 2 (tuner panic
without fusion) is a secondary manifestation of the same autotune machinery and
is folded into §2; Bug 7 (NAFNet fp16 overflow) is inherent to the weights +
input, not an upstream bug — its finding is in `docs/models.md` (Notes).

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
slower. Upstream status: still present on `main` (2026-08-18), only the message
got more verbose.

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
the server invalid ("Memory page 0 doesn't exist"), surfacing as the tuner panic
(Bug 2) and/or the ordering panic (Bug 1). Workaround: the tiled path (640 px)
avoids the 4M-column matmul. No upstream fix. Related: `#1384` closed, `#1401`
CUDA.

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
the state dict with `{k: v.contiguous() for k, v in sd.items()}`.

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
