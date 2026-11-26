# FasterGS2D

FasterGS2D trains oriented disks with two tangent scales per primitive. The
CUDA rasterizer returns RGB, alpha, depth, world-space normals and distortion.
Training selects the buffers and gradients required by the active losses.
The Python training recipe is adapted from MIT-licensed NeRF-ICG revision
57ddd49, copyright 2026 Moritz Kappel, Florian Hahlbohm and Timon Scholz.
The independent disk rasterizer and its numerical checks live in FasterGS.
Point-cloud initialization uses scikit-learn nearest neighbors.

## Run Garden

From the NeRF-ICG root, clone FasterGS into the methods directory and build
its CUDA extension in the NeRF-ICG environment. Place Garden at
`dataset/garden` or set `DATASET.PATH` in the config.

```sh
git clone https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
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

## Run Tanks and Temples

The COLMAP recipe defaults to Barn at `dataset/tnt/Barn`, uses half-resolution
images and supervises normals with median depth. From the NeRF-ICG root run

```sh
python scripts/train.py -c configs/fastergs2d_tnt.yaml
```

## Validation and scope

Run the headless checks from the framework root. The SDL setting keeps GUI
installations from invoking the test runner again on exit.

```sh
SDL_MAIN_HANDLED=1 PYTHONPATH=src python -m unittest discover -s src/Methods/FasterGS/tests
```

For synthetic GPU integration, copy the smoke YAML, set
`TRAINING.DENSIFICATION_GRAD_THRESHOLD` to 0.1 and set
`TRAINING.LOSS` to `{LAMBDA_DISTORTION: 0.1, LAMBDA_NORMAL: 0.05}`.
Pass that file to `python src/Methods/FasterGS/tests/check_fastergs2d.py --config CONFIG --output RESULT.json`.

Depth blending uses `RENDERER.DEPTH_RATIO`. The GUI can display world-space
normals. MCMC, ABSgrad, 3D filters, Speedy-Splat scoring and PPISP are
unsupported for disks. For training benchmarks, numerical checks and colored
TSDF mesh extraction, see `src/Methods/FasterGS/docs/disks.md`
in your FasterGS checkout.
