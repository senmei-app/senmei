# Upstream bug reports (ready to paste)

Draft texts for the burn/cubecl issues tracked in `docs/burn-bugs.md`.
Written for upstream (English). Copy each block as-is into the linked issue
or new-issue form.

---

## 1. burn-fusion "Ordering is bigger than operations" — comment on tracel-ai/burn#4950

Link: <https://github.com/tracel-ai/burn/issues/4950> (same panic, thin report).
Post the following as a comment.

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

## 2. cubecl-wgpu: autotune OOMs the device on an oversized matmul, then corrupts the server

New issue in `tracel-ai/cubecl`. Links: #1384, #1401 (same stale-page symptom,
different trigger).

Title:

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

## 3. burn-nn GroupNorm: f16 div_scalar underflows for large per-group element counts

New issue in `tracel-ai/burn`. No existing report.

Title:

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

Reproducer (f16): the sum of 65536 elements of 0.5 is 32768 (correct), but
sum_dim(2).div_scalar(65536.0) returns 0.0, while mean_dim(2) returns 0.5.

Real case: RealPLKSR's GroupNorm(4, 64) on 64×64 maps → 65536/group; the
normalized output was ±318 vs the torch reference ±1.3.

**Expected**
div_scalar should not underflow on f16 — either promote the reciprocal to f32,
or compute the mean with a scaled kernel like mean_dim.

**Workaround**
Compute the normalization with mean_dim instead of sum + div_scalar.
```

---

## 4. burn-store PytorchReader ignores tensor strides (non-contiguous .pth loads scrambled)

New issue in `tracel-ai/burn`. No existing report.

Title:

```text
burn-store PytorchReader ignores tensor strides (non-contiguous .pth loads scrambled)
```

Body:

```text
**Describe the bug**
PytorchReader parses the stride tuple (args[3] in rebuild_tensor_v2) but discards
it, reading every tensor's storage linearly. A .pth whose weights were saved
non-contiguous therefore loads silently scrambled — the model runs but produces
garbage.

Real case: RealPLKSR 4x_Alchemy.pth stores conv weights channels-last — shape
[o,i,k,k] with strides (27,1,9,3). The loaded head weights had correlation 0.24
vs the correct tensor.

**Expected**
Capture the stride and scatter the storage into the logical layout (or
.contiguous() the data before returning), matching torch's semantics.

**Workaround**
Pre-process the state dict with {k: v.contiguous() for k, v in sd.items()}.
```
