# FasterGS2D

FasterGS2D trains oriented disks with two tangent scales per primitive. The
CUDA rasterizer returns RGB, alpha, depth, world-space normals and distortion.
Training selects the buffers and gradients required by the active losses.
The Python training recipe is adapted from MIT-licensed NeRF-ICG revision
57ddd49, copyright 2026 Moritz Kappel, Florian Hahlbohm and Timon Scholz.
The independent disk rasterizer and its numerical checks live in FasterGS.
Point-cloud initialization uses scikit-learn nearest neighbors.

## Run Garden

Clone FasterGS into the methods directory and build its CUDA extension in the
NeRF-ICG environment. The commands below use the paired `2DGS_ready` branches.
Place Garden at `dataset/garden` or override the dataset path on the command
line. From the NeRF-ICG root run

```sh
git clone -b 2DGS_ready https://github.com/AlexMorgand/faster-gaussian-splatting.git src/Methods/FasterGS
python -m pip install --no-build-isolation ./src/Methods/FasterGS/FasterGSCudaBackend
python scripts/train.py -c configs/fastergs2d_smoke.yaml
python scripts/train.py -c configs/fastergs2d_garden.yaml
```

The smoke config runs 20 iterations. The 30,000-iteration Garden config uses
quarter-resolution images and holds out every eighth view. Normal loss has
weight 0.05 after iteration 7000. Distortion has weight zero. Densification
sums the image-shift gradient norm in normalized device coordinates over
accepted views and divides by the number of views with projected support.
The example threshold was calibrated for this Garden recipe.

A development run on one RTX 5090 took 351.8 seconds for the trainer, after
image prefetch, and finished with 2,955,153 disks. Across 24 held-out views
it scored 27.000 dB PSNR, 0.8554 SSIM and 0.1244 VGG-LPIPS.

## Run Tanks and Temples

The COLMAP recipe defaults to Barn at `dataset/tnt/Barn`, uses half-resolution
images and supervises normals with median depth. From the NeRF-ICG root run

```sh
python scripts/train.py -c configs/fastergs2d_tnt.yaml
```

## Validation and scope

Run `python -m unittest discover -s src/Methods/FasterGS/tests` with `src` on PYTHONPATH.
For synthetic GPU integration, copy the smoke YAML, set
`TRAINING.DENSIFICATION_GRAD_THRESHOLD` to 0.1 and set
`TRAINING.LOSS` to `{LAMBDA_DISTORTION: 0.1, LAMBDA_NORMAL: 0.05}`.
Pass that file to `python src/Methods/FasterGS/tests/check_fastergs2d.py --config CONFIG --output RESULT.json`.
See the FasterGS checkout's `docs/disks.md` for native numerical checks.

Depth blending uses `RENDERER.DEPTH_RATIO`. The GUI can display world-space
normals. MCMC, ABSgrad, 3D filters, Speedy-Splat scoring and PPISP are
unsupported for disks. Mesh extraction is outside the scope of this method.
