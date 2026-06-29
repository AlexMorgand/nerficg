# Native 2DGS surfel CUDA rasterizer

Perspective-correct surfel rasterization for Faster2DGS, vendored from the
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) /
[diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization)
reference implementation (Inria GRAPHDECO, non-commercial license — see
`submodules/diff-surfel-rasterization/LICENSE.md`).

Built as part of `Faster2DGSCudaBackend._C` — no separate pip package required.

Outputs:
- RGB (3 × H × W)
- Allmap (7 × H × W): expected depth, alpha, view-space normal, median depth, distortion

GLM headers are taken from `submodules/diff-surfel-rasterization/third_party/glm`
at build time (`git submodule update --init`).
