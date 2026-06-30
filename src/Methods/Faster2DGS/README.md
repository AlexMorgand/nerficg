# Faster2DGS

NeRFICG integration of [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) with a **native CUDA surfel rasterizer** (`Faster2DGSCudaBackend`). Training loop, losses, and densification match official `train.py`; rendering goes through the in-tree surfel kernel (photometric fast paths optional).

## Install

```bash
python scripts/install.py -m Faster2DGS
```

## MipNeRF360 (official m360 defaults)

Use `configs/2DGS_m360.yaml` — same as official 2DGS `m360_eval.py` (`λ_dist=0`, `depth_ratio=0`, no PCA). Only image resolution differs:

| Scenes | Folder | `IMAGE_SCALE_FACTOR` |
|--------|--------|----------------------|
| outdoor (garden, stump, bicycle) | `images_4` | `0.25` |
| indoor (kitchen, room, counter, bonsai) | `images_2` | `0.5` |

**Train @30k (paper benchmark):**

```bash
python scripts/train.py -c configs/2DGS_m360.yaml \
    DATASET.PATH=dataset/mipnerf360/garden DATASET.IMAGE_SCALE_FACTOR=0.25

python scripts/train.py -c configs/2DGS_m360.yaml \
    DATASET.PATH=dataset/mipnerf360/kitchen DATASET.IMAGE_SCALE_FACTOR=0.5
```

**Train + test PSNR in one shot:**

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

Bounded TSDF export aligned with official 2DGS (`scripts/faster2dgs_render_mesh.py`):

```bash
python scripts/faster2dgs_render_mesh.py -c configs/2DGS_m360.yaml \
    DATASET.PATH=dataset/mipnerf360/garden DATASET.IMAGE_SCALE_FACTOR=0.25 \
    -d output/Faster2DGS/<run_dir>
```

Buffer sanity check: `scripts/faster2dgs_sanity_buffers.py`.

## Speed / profiling

- `scripts/faster2dgs_speed_experiment.py` — throughput experiments
- `scripts/faster2dgs_profile_step.py` — per-step CUDA timing + densify log
- `scripts/faster2dgs_densify_compare.py` — densify milestone breakdown

## Next steps

With training parity established, focus areas:

1. **Beyond PSNR** — SSIM/LPIPS tables, normal maps, depth, mesh Chamfer vs official
2. **Meshing** — unbounded extraction, cluster post-process tuning
3. **Speed** — photometric-only backward fast path (already in kernel), batching, memory (`PRELOADING_LEVEL`)

## Layout

| Path | Role |
|------|------|
| `Trainer.py` | Official 2DGS loop order, geometry loss gates |
| `Gaussians2D.py` | Surfels, densify/prune (official-compatible) |
| `Renderer.py` | Surfel raster + auxiliary maps |
| `mesh_utils.py` | TSDF / post-process helpers |
| `Faster2DGSCudaBackend/` | Native surfel CUDA rasterizer |
