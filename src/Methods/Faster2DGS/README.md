# Faster2DGS

NeRFICG integration of [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) with a **native CUDA surfel rasterizer** (`Faster2DGSCudaBackend`). Training loop, losses, and densification match official `train.py`; rendering goes through the in-tree surfel kernel (photometric fast paths optional).

## Install

```bash
python scripts/install.py -m Faster2DGS
```

## Two configs

| Config | Goal | Key settings |
|--------|------|----------------|
| **`configs/2DGS_m360.yaml`** | Photometric / M360 PSNR parity | `λ_dist=0`, `depth_ratio=0` |
| **`configs/2DGS_mesh.yaml`** | Mesh + geometry (COLMAP, bounded indoor) | `λ_dist=1000`, `depth_ratio=1`, normal @7k |

Override `DATASET.PATH`, `GLOBAL.DATASET_TYPE`, and scale on the CLI — no per-scene yaml files.

**Photometric (novel-view RGB, benchmarks):**

```bash
python scripts/train.py -c configs/2DGS_m360.yaml \
    DATASET.PATH=dataset/mipnerf360/garden DATASET.IMAGE_SCALE_FACTOR=0.25

# Custom COLMAP
python scripts/train.py -c configs/2DGS_m360.yaml \
    GLOBAL.DATASET_TYPE=Colmap \
    'DATASET.PATH=livingroom_take2/undistort' DATASET.IMAGE_SCALE_FACTOR=0.5
```

**Mesh (TSDF export, surface-regularized training):**

```bash
python scripts/train.py -c configs/2DGS_mesh.yaml \
    'DATASET.PATH=livingroom_take2/undistort' DATASET.IMAGE_SCALE_FACTOR=0.5

python scripts/faster2dgs_render_mesh.py -d output/Faster2DGS/<run_dir>
```

Outdoor mesh overrides on `2DGS_mesh.yaml`: `RENDERER.DEPTH_RATIO=0` `TRAINING.LOSS.LAMBDA_DISTORTION=100` `DATASET.IMAGE_SCALE_FACTOR=0.25`.

MipNeRF360 resolution (both configs):

| Scenes | Folder | `IMAGE_SCALE_FACTOR` |
|--------|--------|----------------------|
| outdoor (garden, stump, bicycle) | `images_4` | `0.25` |
| indoor (kitchen, room, counter, bonsai) | `images_2` | `0.5` |

**Train + test PSNR in one shot (photometric):**

```bash
cd scripts
PRELOADING_LEVEL=0 python faster2dgs_eval.py -c ../configs/2DGS_m360.yaml \
    DATASET.PATH=../dataset/mipnerf360/garden DATASET.IMAGE_SCALE_FACTOR=0.25 \
    TRAINING.MODEL_NAME=garden --iters 30000
```

## Validated quality (Jun 2026)

Photometric @7k (`λ_dist=0`, normal off) vs official 2DGS — all seven M360 scenes within **±0.4 dB**:

| Scene | Faster2DGS | Official | Δ PSNR |
|-------|------------|----------|--------|
| kitchen | 28.50 | 28.88 | −0.38 |
| bonsai | 29.38 | 29.30 | +0.08 |
| room | 29.22 | 29.10 | +0.12 |
| counter | 26.87 | 26.90 | −0.03 |
| garden | 25.85 | 25.83 | +0.02 |
| bicycle | 22.98 | 22.96 | +0.03 |
| stump | 25.17 | 25.31 | −0.14 |

Full recipe @30k — **garden: 26.47 PSNR** (paper 26.95, −0.48 dB) with normal/distortion active.

Reproduce parity:

```bash
cd scripts
PRELOADING_LEVEL=0 ./faster2dgs_parity_7k_batch.sh --all-remaining   # one scene at a time, flock lock
python faster2dgs_parity_7k.py --scene kitchen
```

## Meshing

Train with **`configs/2DGS_mesh.yaml`** first (not `2DGS_m360`). Then bounded TSDF export:

```bash
python scripts/faster2dgs_render_mesh.py -d output/Faster2DGS/<run_dir> \
    'DATASET.PATH=livingroom_take2/undistort' DATASET.IMAGE_SCALE_FACTOR=0.5
```

Pass dataset overrides if `training_config.yaml` in the run dir has the wrong path. Buffer sanity: `scripts/faster2dgs_sanity_buffers.py`.

**External masks (COLMAP):** set `DATASET.EXTERNAL_MASKS_PATH` to a folder of per-image masks (matched to `images/` filenames). Use `EXTERNAL_MASKS_BINARY: false` for soft ViTMatte masks. Masks apply to training loss compositing and TSDF depth masking (default on). Legacy `TRAINING.EXTERNAL_MASKS_PATH` is forwarded to `DATASET` at startup.

**Depth filtering (bounded TSDF, 2DGS-aligned + extras):**

| Stage | What it does |
|-------|----------------|
| GT / rendered alpha mask | Zero background depth (2DGS `gt_alpha_mask`) |
| `depth_trunc` | Camera-space Z cap (`2 × scene radius` auto, or `--depth_trunc`) |
| Scene-sphere filter (default on) | Zero pixels whose backprojection lies outside `sphere_scale × radius` from the focus point |
| Depth percentile prescan (auto trunc) | Tighten `depth_trunc` using p99 of rendered depths (`--depth_percentile 0.99`) |

If the mesh is still too large or has distant hulls: lower `--sphere_scale` (e.g. `1.5`) or set an explicit `--depth_trunc`. If the mesh is empty: increase `--depth_trunc` or pass `--no_sphere_filter`.

## Speed / profiling

Baseline workflow (run from ``scripts/``):

```bash
# Per-component step breakdown (photometric @2k vs full-aux @3.1k)
PRELOADING_LEVEL=0 python faster2dgs_profile_step.py -c ../configs/2DGS_m360.yaml \
    DATASET.PATH=../dataset/mipnerf360/stump DATASET.IMAGE_SCALE_FACTOR=0.25 \
    --iters 3500 --profile-at 2000 3100 --repeats 30

# Kernel fwd+bwd: photometric vs full aux
python faster2dgs_speed_experiment.py --profile-at 2000 3100 --repeats 30

# Surfel vs FastGS 3D kernel at matched splat count
python rasterizer_kernel_compare.py --advance-to 3100 --method both
```

Scripts:

| Script | Purpose |
|--------|---------|
| `faster2dgs_profile_step.py` | Per-step CUDA timing + densify log |
| `faster2dgs_speed_experiment.py` | Photometric vs full-aux kernel throughput |
| `rasterizer_kernel_compare.py` | Surfel vs FastGS 3D isolated kernel |
| `faster2dgs_densify_compare.py` | Splat milestones vs official |

### Speed roadmap (FastGS-style)

Already inherited from FastGS: **FusedAdam**, **fused DSSIM**, **Morton reordering**, **expandable segments**, **32-bucket parallel backward**, **photometric fast path** (iter ≤3k).

Recent wins:

- **`render_image_benchmark()`** — RGB-only inference (photometric kernel, no autograd) for eval throughput
- **Lazy `depth_to_normal`** — skipped during distortion-only phase (iter 3k–7k)

Priority levers (biggest training-step impact first):

| Priority | Lever | Status | Notes |
|----------|-------|--------|-------|
| P0 | Surfel **inference kernel** (no bucket save) | partial | benchmark path done; dedicated `inference.cu` like FastGS next |
| P0 | **Distortion-only CUDA aux** | done | iter 3k–7k skips normal/median fwd+bwd |
| P1 | **CUDA densification stats** | todo | replace screenspace autograd hook with in-kernel accum (FastGS pattern) |
| P1 | **Fused backward + FusedAdam** | todo | port FasterGSFused branch to surfel backend |
| P2 | **Splat budget / Speedy-Splat** | optional | `USE_SPLAT_BUDGET` exists; surfel pruning kernel TBD |
| P2 | Densify hygiene | todo | drop redundant `empty_cache()` / sync in `_run_densify` |

**Regression guard:** keep PSNR parity (`faster2dgs_parity_7k.py`); track SSIM/LPIPS via `faster2dgs_eval.py` on full @30k runs.

**Timings in testing:** eval and parity runs write `timings.txt` (wall clock + ms/iter per callback) and optional `timing_summary.json`. Use `--profile-default` on eval/parity for one end-of-run CUDA step benchmark; mid-train checkpoints via `faster2dgs_profile_step.py --advance-to 3100`.

## Next steps

1. **Speed** — P0 distortion-only kernel + dedicated inference CUDA (target: match FastGS step rate at equal splat count)
2. **Metrics** — SSIM/LPIPS tables @30k; normal/depth visual compare vs official
3. **Meshing** — unbounded extraction, cluster post-process tuning

## Layout

| Path | Role |
|------|------|
| `Trainer.py` | Official 2DGS loop order, geometry loss gates |
| `Gaussians2D.py` | Surfels, densify/prune (official-compatible) |
| `Renderer.py` | Surfel raster + auxiliary maps |
| `mesh_utils.py` | TSDF / post-process helpers |
| `Faster2DGSCudaBackend/` | Native surfel CUDA rasterizer |
