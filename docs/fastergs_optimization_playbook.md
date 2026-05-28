# FasterGS Optimization Playbook (Turntable + Masked Metrics)

This playbook implements the ablation-first plan for FasterGS optimization.

## 1) Evaluation Gate (must pass first)

### A. Validate scene/mask filename alignment for masked metrics

```bash
python scripts/fastergs_eval_gate.py --results-root /path/to/eval_root
```

Expected structure per scene:

- `gt/` with ground-truth images
- method folders (same filenames as `gt/`)
- optional `_mask/` (same filenames as `gt/`)

The check fails fast if filenames do not align, to avoid silent metric corruption.

### B. Validate turntable train-vs-infer parity for a checkpoint

```bash
python scripts/fastergs_eval_gate.py \
  --training-dir /path/to/output/your_run \
  --checkpoint final.pt \
  --subset test \
  --max-views 8
```

This compares:

- training path render (`render_image_training`) and
- inference path render (`render_image_inference`)

for turntable views, and reports MAE / max absolute error.

## 2) Generate and run prioritized ablations

Use your current base configs and generate the standard ablation set:

```bash
python scripts/fastergs_ablation_runner.py \
  --base-config gs_generic.yaml \
  --output-dir configs/ablation_generic \
  --mode ablation \
  --short-iters 12000
```

```bash
python scripts/fastergs_ablation_runner.py \
  --base-config gs_prior.yaml \
  --output-dir configs/ablation_prior \
  --mode ablation \
  --short-iters 12000
```

Run all generated configs sequentially:

```bash
python scripts/fastergs_ablation_runner.py \
  --base-config gs_prior.yaml \
  --output-dir output/ablation_configs \
  --mode ablation \
  --short-iters 12000 \
  --run
```

Generated variants:

- `mesh_locked`
- `mesh_selective_unfreeze`
- `mesh_unfreeze_densify`
- `pointcloud_adc`
- `pointcloud_mcmc`
- `mesh_mcmc_unfreeze`

## 3) Run stress-test set on top candidates

Generate stress variants:

```bash
python scripts/fastergs_ablation_runner.py \
  --base-config gs_prior.yaml \
  --output-dir configs/stress_prior \
  --mode stress \
  --short-iters 12000
```

Stress variants include reduced-view and turntable reference-frame sensitivity.

## 4) Metrics reporting policy

- Primary: `mPSNR`, `mSSIM`, `mLPIPS` via `scripts/generate_tables.py`.
- Secondary: full-frame `PSNR`, `SSIM`, `LPIPS_*`.
- Always keep `_mask` filenames aligned with `gt`.
- Set `TRAINING.EXTERNAL_MASKS_BINARY: false` for grayscale masks; keep it `true` for thresholded binary masks.

### Ranking runs from metrics files

After test renders are produced (each run writing `metrics_8bit.txt`), rank all runs:

```bash
python scripts/rank_metrics.py --dir output
```

This writes `output/metrics_ranking.txt` with:

- overall ranking across common metrics (average rank),
- mipnerf-style composite ranking (when `PSNR`, `SSIM`, `LPIPS` exist),
- ranking per individual metric.

## 5) Presets

- `configs/fastergs_turntable_prod_masked.yaml`:
  stable production default (mesh-prior preserving).
- `configs/fastergs_turntable_specular_friendly.yaml`:
  more adaptive setting for difficult specular/highlight cases.
- `gs_generic.yaml`:
  unbiased baseline for point-cloud-first reconstruction.
- `gs_prior.yaml`:
  prior-mesh optimized baseline for turntable captures.
