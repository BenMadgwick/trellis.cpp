# Findings, 2026-09-04 — measured behaviour of the offramp and cache paths

Measured while adding the resume cache, the multi-target face sweep and the
per-stage RNG. Everything here is measured rather than estimated, and several
items correct earlier ESTIMATES that were never measured.

All numbers: container_briefcase, seed 42, default steps, RTX 3080 Laptop,
sm_86 dev build, machine otherwise idle.

---

## 1. Gating costs ~9 s, not ~180 s

| run | time |
|---|---|
| straight through, **cold** file cache | 381.7 s |
| straight through, **warm** | **328.0 s** |
| gate (`--vox-only --save-vox --vox-render`) | 33.9 s |
| resume (`--load-vox`) after accepting | 302.7 s |
| gate + resume | 336.6 s |

**Overhead of splitting = 336.6 − 328.0 = 8.6 s (2.6%).** That is process
restart, CUDA re-init, and a 20 MB `.vox` write plus read.

**Corrects an earlier estimate**, which put preview-then-accept at ~180 s extra on
an accepted job and concluded it "pays off above roughly a 40% reject rate".
At 2.6% the gate is worth having even at a near-zero reject rate.

### The "~55 s fixed overhead" is a cold-cache constant

An earlier note described ~55 s of fixed overhead that "does NOT shrink with
steps". The cold-to-warm gap here is **53.7 s**. That strongly suggests the
figure was measuring the first read of 6.3 GB of GGUF weights from disk, not a
per-run cost: on a busy shared machine the weights stay in the OS file cache and
it is not paid at all. An ETA model should not carry it as a constant.

---

## 2. A resumed run is NOT the same asset as a straight run

Same image, same `--seed 42`:

| | straight | resumed |
|---|---|---|
| active voxels @res32 | 2,735 | 2,735 |
| conditioning tokens | 1029 / 4101 | 1029 / 4101 |
| **upsampled coords @res512** | **769,110** | **783,297** |
| HR tokens | 11,211 | 11,225 |
| final GLB | 10.33 MB | 11.09 MB |

The structure is bit-identical — the `.vox` cache is faithful and the preview
genuinely shows what was approved. Everything downstream diverges, because the
run uses ONE `std::mt19937` stream: a straight run draws noise during the
structure flow, a resumed run skips those stages, so the shape flow begins at a
different position in the stream.

**Consequences for a content-addressed cache**

- A cache HIT (resume) and a cache MISS (straight) produce different assets from
  identical inputs. Mixed paths mean two users with the same image and seed get
  different geometry — the "somebody else's asset" failure wearing a disguise.
- `--seed 42` straight != `--seed 42` resumed. Seed reproducibility is weaker
  than it looks.

**Fix:** seed each flow from `hash(run_seed, stage)` rather than sharing one
stream. Straight and resumed then agree bit-for-bit, re-rolling is unaffected,
and the cache becomes transparent. This is a prerequisite for the cache being
correct, not an enhancement.

Note this is NOT addressed by the v2 cache's LR sharing (§2.3): that only helps
when the cached run actually ran the LR pass, and `--vox-only` stops before it.

---

## 2b. A low-step preview is faithful ONLY when the structure is shared

An earlier note recommended `--steps 4` for previews on the strength of a
table showing HR tokens within 0.81% of the 12-step value. That table was taken
**"same seed, same cache"** — i.e. resumed, with the voxel structure loaded and
held fixed — so it measured what `steps` does to the shape and texture flows
only. It never measured the structure flow.

Measured here on the briefcase, straight (no cache), only `steps` varying:

| steps | active voxels @res32 |
|---|---|
| 4 | 1,611 |
| 12 | 2,735 |

**41% fewer voxels.** `--steps` drives the sparse-structure flow too, so a
straight low-step run generates a genuinely different structure — not a coarser
rendering of the same one.

Consequences:

- The gate must run at the **same step count as the final run**, which for the
  agreed design means the default. It already does; the risk is somebody later
  "optimising" the gate to `--steps 4` and silently making the preview a
  prediction of an asset nobody is going to build.
- `--steps` as a cheap preview knob is only sound *downstream of a shared
  structure*, i.e. on the `--load-vox` path. That is exactly where the earlier
  table was measured, which is why the table is right and the recommendation
  drawn from it is not.

---

## 3. The GPU decimator is non-deterministic run to run

Two straight runs, same seed, identical through the ENTIRE remesh — same
upsampled coords, same `remesh_dc` face counts, same dropped crossing quads,
same parity counts — then:

```
decimate_qem_gpu(target=300000): V 2966111->147910, F 5931444->295516
decimate_qem_gpu(target=300000): V 2966111->147906, F 5931444->295492
```

Identical input, 24 faces different (0.008%), and the divergence cascades into
different UVs and a ~4% different GLB size. Almost certainly non-deterministic
reduction ordering in the CUDA kernel.

**Consequences**

- Never verify a cache entry by hashing the output GLB; it will not match itself.
- Key the cache on INPUTS only and treat the stored artifact as authoritative.
- Everything up to and including the remesh IS reproducible, which makes it the
  right place to A/B a change to the mesh stage.

---

## 4. The preview renderer could not show a dark asset

Fixed in `a28d372`. The renderer multiplied albedo by the shading term and drew
onto a near-black ground (24/255), so a black briefcase rendered at **16/255
across every face** — one flat tone, no form — on a background of 24. Its actual
fault (a large raised blob on the front face, the model reading a shading
gradient in the source leather as geometry) was invisible in the coloured
turntable and only appeared in a GLB render.

More ambient could not have fixed it: the shading term is MULTIPLICATIVE, so an
albedo of 0 stays 0 however the scene is lit. The fix adds `ALBEDO_LIFT` (45)
to the base colour BEFORE shading and puts the background on a mid-grey vertical
ramp (128 -> 98). A black asset now spans 22..61 against that ground.

Verified: the blob and the centre seam are both plainly visible in the
regenerated turntable.

---

## 5. Three different meanings of the "post" dump

Same binary layout, three different contents:

| producer / consumer | what the mesh is |
|---|---|
| env `TRELLIS_DUMP_POST` | **pre-weld** raw decoded mesh |
| CLI `--dump-post` (upstream, merged in v0.7.0) | **post-remesh** cleaned mesh |
| `post-replay` (the consumer) | expects **pre-remesh** — it welds, dedupes, builds a BVH and remeshes what it loads |

So `trellis-cli --dump-post X` followed by `post-replay X` **double-remeshes**,
offsetting an already-offset surface. Defensible for `--dump-post`'s stated
purpose (QtMeshEditor owns its own processing) but the comment claiming layout
parity with the env var is misleading, and the format cannot be used as a cache
without knowing which producer wrote it.

**The master must hold the PRE-remesh mesh**: the albedo snap samples a BVH over
it, as post-replay's own comment states. Storing only the post-remesh mesh would
silently move texel positions.

---

## 6. Drift found between the two copies of the mesh stage

Removed by the shared `remesh_stage()`:

1. **post-replay's `auto` fallback left `rmode` stale.** On falling back it set
   `do_strip = true` but never updated `rmode`, so later single-cover checks
   could read `Interior` after `Unsigned` had been built. Now fixed by
   construction — the stage returns `final_mode`.
2. **`band` defaults differ**: `1` in post-replay, `0` (auto -> `res/512`, i.e.
   **2** at res 1024) in trellis-cli. A default "replay" therefore contours a
   THINNER shell than the run it replays, which makes any A/B between the two
   partly a band comparison. **Preserved, not changed** — that is a product
   decision, not a refactor's.
3. **Two different open-edge counters** feeding the same threshold. Now one,
   sort-based rather than `unordered_map`, same definition.

---

## 7. Confirmed log markers for the GUI parser

Verified against real output, matching the recorded validation figures exactly:

```
      active voxels @res32 = 2735
      upsampled coords @res512=769110 -> quantized @res1024 (grid 64) = 11211 tokens
      LR slat + upsampled coords reused from the cache (12 steps, 769110 coords)
      [vox] preview -> PATH (1280x640, 8-view turntable at 45 deg, pitched 20 deg down)
      [vox] cache -> PATH (2735 voxels, cond 1029+4101 tokens, 20.0 MB)
[vox-only] done (33.9s) -- resume with --load-vox
[tex-only] done (76.2s) -- stopped before mesh post-processing
```

Two parser notes:

- Flow progress lines carry a live ETA and are **carriage-return separated**
  even when redirected to a file: `[flow] [#####...............]  1/4  6.7s  ~20s left`.
  A tail that splits on `\n` alone will see one enormous line.
- The preview is ONE png, a 4x2 sheet of eight 320 px tiles (1280x640). A
  turntable slider scrubs `background-position`, it does not load eight files.
- Re-rendering a preview from an existing `.vox` costs **0.1 s and no GPU**, so
  previews can be regenerated at other sizes or angles on demand rather than
  stored.
