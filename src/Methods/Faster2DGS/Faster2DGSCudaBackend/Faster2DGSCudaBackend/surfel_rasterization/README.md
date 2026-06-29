# Faster2DGS native surfel CUDA rasterizer

NeRFICG implementation of perspective-correct 2D Gaussian surfel rasterization.
Built as `Faster2DGSCudaBackend._C` — no external rasterizer packages required.

## Architecture

- **FastGS bucket infrastructure**: depth/tile sorting, 32-primitive bucket checkpoints,
  `<<<n_buckets, 32>>>` warp-parallel backward.
- **2DGS surfel math**: tangent-plane transform `T`, ray–splat UV intersection,
  `rho = min(rho3d, rho2d)`, 7-channel allmap (expected depth, alpha, normal,
  median depth, distortion).

## Outputs

- RGB: `(3, H, W)`
- Allmap: `(7, H, W)` — same channel layout as the 2DGS paper training code

## Performance notes

- **Photometric fast path** (iterations before geometry losses): skips aux accumulation
  in forward/backward and Python `depth_to_normal` — no RGB quality impact.
- **Splat budget** (`USE_SPLAT_BUDGET`): **off by default**. When enabled, soft grad
  scaling (`SPLAT_BUDGET_TARGET`) and optional hard prune (`SPLAT_BUDGET_HARD_CAP`) trade
  quality for speed — use only for benchmarking, not production training.
- Do not enable `SURFEL_TILE_CIRCLE_CULL` or `TIGHTBBOX` for production training.

## Build

```bash
python scripts/install.py -m Faster2DGS
```

GLM headers are vendored under `third_party/glm/` (MIT).
