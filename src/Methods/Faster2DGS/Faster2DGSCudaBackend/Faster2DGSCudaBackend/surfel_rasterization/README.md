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

## Build

```bash
python scripts/install.py -m Faster2DGS
```

GLM headers are vendored under `third_party/glm/` (MIT).
