# Faster2DGS Migration Plan (from FastGS/3DGS baseline)

This repository currently has a **FastGS + auxiliary buffers** variant. It is not yet a true 2DGS surfel pipeline.

This document defines the migration target and implementation order.

## 1) Target Differences vs FastGS

### Primitive representation (must change)
- Current FastGS primitive:
  - `mean3d`: `(N, 3)`
  - `log_scale`: `(N, 3)` (full 3D ellipsoid)
  - `quat`: `(N, 4)`
  - `opacity`: `(N, 1)`
  - `sh`: `(N, B, 3)`
- Target 2DGS primitive:
  - `mean3d`: `(N, 3)`
  - `log_scale2d`: `(N, 2)` (disk/tangent plane)
  - `quat`: `(N, 4)` (tangent frame orientation)
  - `opacity`: `(N, 1)`
  - `sh`: `(N, B, 3)`

### Rasterization (must change)
- Current: 3DGS-style EWA splatting with added aux channels.
- Target: perspective-correct surfel rasterization (2DGS / diff-surfel style):
  - Tangent-plane transform per primitive.
  - Pixel-surfel intersection in local UV.
  - Surfel footprint/rho evaluation.

### Gradients (must change)
- Current: aux gradients approximated by folding into RGB gradient.
- Target: analytic backward over color + aux outputs:
  - Backward API must accept `grad_color` and `grad_aux`.
  - Return gradients for means, scale2d, rotation, opacity, SH.

### Losses (must expand)
- Keep: photometric (L1 + DSSIM), normal consistency, distortion.
- Add: depth smoothness, surface compactness, multi-view consistency (phased).


## 2) Code Migration Phases

## Phase A - Add dedicated 2DGS backend module (no model breakage yet)
1. Create `src/Methods/Faster2DGS/Faster2DGSCudaBackend/` with extension entrypoint.
2. Expose Python bindings in `torch_bindings/rasterization.py`:
   - `RasterizerSettings`
   - `diff_rasterize_surfel(...) -> (rgb, aux)`
   - autograd `backward(grad_rgb, grad_aux)`
3. Do not reuse FastGS kernel signatures. Keep separate C++ API.

Acceptance:
- Importing `Methods.Faster2DGS.Faster2DGSCudaBackend` works.
- A unit smoke test can call forward and backward with tiny tensors.

## Phase B - Introduce 2DGS primitive storage
1. Replace `Faster2DGSModel(FasterGSModel)` inheritance with dedicated model class.
2. Implement a `Gaussians2D` module:
   - Parameters: means, sh0, sh_rest, raw_opacities, raw_scales_2d, raw_rotations.
   - Required trainer methods: init, training_setup, optimizer state updates, SH progression.
3. Keep densification initially conservative:
   - Clone/split logic adapted for 2D scales.
   - No FastGS-specific 3D filter assumptions.

**Status (implemented):** `Gaussians2D` stores `(N, 2)` log-scales; 2D-aware densification; PLY exports `scale_0/scale_1`; legacy `(N, 3)` checkpoints load by keeping XY only. Rasterization still uses the Phase A surfel bridge (Phase C).

Acceptance:
- Training loop runs for 100+ iterations with `METHOD_TYPE=Faster2DGS` and no shape hacks.

## Phase C - Wire renderer to dedicated surfel kernel
1. Update `src/Methods/Faster2DGS/Renderer.py` to call new backend only.
2. Remove legacy fallback that interprets FastGS aux as surfel output.
3. Standardize aux channel layout (fixed contract):
   - `aux[0]`: expected depth numerator
   - `aux[1]`: alpha
   - `aux[2:5]`: rendered normal (view space → world in renderer)
   - `aux[5]`: median depth
   - `aux[6]`: distortion

**Status (implemented):** Native surfel CUDA kernels in
`Faster2DGSCudaBackend/surfel_rasterization/` and built with
`python scripts/install.py -m Faster2DGS`. No external rasterizer pip package.
Old FastGS 3D bridge removed.

Acceptance:
- `render_image_inference` returns non-degenerate `rgb/depth/alpha/normal` on trained checkpoint.

## Phase D - Loss alignment
1. Keep existing:
   - `L1 + DSSIM`
   - normal consistency
   - distortion
2. Add phased terms:
   - depth smoothness (edge-aware on `surf_depth`, optional `LAMBDA_DEPTH_SMOOTHNESS`)
   - surface compactness (via distortion loss `L_d` — paper α=100/1000)
   - multi-view consistency (not yet)
3. Ensure these terms flow through analytic aux backward.

**Status (implemented):** Normal + distortion match 2DGS `train.py`; iter gates 3k/7k (binary, no warmup); densify after backward; geometry loss console log every `GEOMETRY_LOG_INTERVAL`; depth smoothness (Phase D); configs use paper α (100 garden / 1000 guitar) + `LAMBDA_NORMAL=0.05`.

Acceptance:
- Loss terms are non-zero where expected and decrease during training.

## Phase E - Meshing + viewer behavior
1. Keep TSDF export script expecting surfel depth/alpha/normal.
2. Mark viewer limitations:
   - Existing 3DGS-style viewers may still display ellipsoids from exported PLY.
3. Add optional 2DGS-aware export path/comments and viewer notes.

**Status (partial):** Mesh export works; viewer still 3DGS-style ellipsoids from PLY.


## Phase F - MipNeRF360 paper parity benchmark

Configs (outdoor `DEPTH_RATIO=0`, `LAMBDA_DISTORTION=100`, `images_4` / factor 0.25;
indoor kitchen: `DEPTH_RATIO=1`, `LAMBDA_DISTORTION=1000`, `images_2` / factor 0.5):

| Scene | Config | Paper PSNR |
|-------|--------|------------|
| bicycle | `configs/gs_bicycle_2DGS.yaml` | 24.87 |
| stump | `configs/gs_stump_2DGS.yaml` | 26.47 |
| kitchen | `configs/gs_kitchen_2DGS.yaml` | 30.50 |
| garden | `configs/gs_garden_2DGS.yaml` | 26.95 |

Run all four sequentially:

```bash
python scripts/run_mip360_2dgs_benchmark.py
```

FastGS speed path retained: `FusedAdam`, `fused_dssim`, Morton reordering (5k),
`expandable_segments`, `PRELOADING_LEVEL=1`, native Faster2DGSCudaBackend surfel rasterizer.

2DGS parity fixes: official viewspace grad for densify (no pixel scale), opacity cull 0.05,
screen-size prune, no final opacity prune, native surfel backend (7-ch allmap).


## 3) Tensor Contracts to Lock Before Coding

These should be decided once and not changed mid-port:

1. **Normal space**: choose world-space or view-space for aux normal output.
2. **Depth definition**: expected depth, median depth, or both.
3. **Scale activation**: `scale2d = exp(raw_scale2d)` (recommended).
4. **Rotation activation**: normalized quaternion.
5. **Aux layout**: fixed channel order for Python/CUDA boundary.


## 4) Non-goals During Initial Port

- Perfect parity with all official 2DGS regularizers in first pass.
- FastGS-specific speed tricks (bucket sorting variants, MCMC extras) in first pass.
- Full viewer replacement.


## 5) Immediate Next Coding Task

Implement **Phase A** (new backend skeleton + forward/backward interface) and switch `Faster2DGSRenderer` behind a feature flag to use it when available.


## Phase G - Native Fast2DGS surfel kernels (NeRFICG, FastGS bucket infra)

**Status (implemented).** Faster2DGS uses a single in-repo CUDA backend
(`Faster2DGSCudaBackend._C`). No external `diff-surfel-rasterization` pip package,
no FasterGS 3D bridge fallback, and no `DiffSurfelBackend` adapter.

What was kept from FastGS:
- 32-primitive **bucket checkpointing** and the `<<<n_buckets, 32>>>`
  warp-parallel backward.
- Buffer/blob allocation pattern (`GeometryState` / `BinningState` /
  `BucketState`), `cub` depth+tile sorting, per-tile instance ranges.

What is 2DGS surfel math (NeRFICG implementation):
- Preprocess: `compute_transmat` (tangent-plane `T`, 9 floats) + `compute_aabb`
  + view-space normal; depth key = `p_view.z`.
- Blend: ray-splat UV intersection, `rho = min(rho3d, rho2d)`, 7-channel allmap
  (expected depth, alpha, normal, median depth, distortion).
- Analytic aux backward folded into the bucket-parallel pass via front-to-back
  reconstruction from bucket checkpoints (`color, T, D, N, M1, M2`).

Build: `python scripts/install.py -m Faster2DGS` (GLM vendored under
`Faster2DGSCudaBackend/third_party/glm`).

Validation: `scripts/faster2dgs_kernel_parity.py` (native smoke + timing).

Acceptance checklist (plan Phase 6):
- Kernel fwd+bwd smoke on stump (32k splats): no NaN, gradients non-zero.
- Training: `python scripts/train.py -c configs/gs_stump_2DGS.yaml -s` with native backend log line.
- Paper parity: `python scripts/run_mip360_2dgs_benchmark.py` (bicycle 24.87 / stump 26.47 / kitchen 30.50 / garden 26.95 PSNR targets).
