#!/usr/bin/env python3
"""Train Faster2DGS to N iterations, render test set, report PSNR vs 2DGS paper.

Examples:
    python faster2dgs_eval.py -c configs/2DGS_m360.yaml \\
        DATASET.PATH=dataset/mipnerf360/garden DATASET.IMAGE_SCALE_FACTOR=0.25 --iters 30000

    python faster2dgs_eval.py -c configs/2DGS_m360.yaml \\
        DATASET.PATH=dataset/mipnerf360/kitchen DATASET.IMAGE_SCALE_FACTOR=0.5 --iters 7000

    # Re-render metrics from an existing run (no training):
    python faster2dgs_eval.py --render-only -d output/Faster2DGS/kitchen_... --checkpoint final.pt

    # Train + end-of-run step timing (~15 CUDA repeats):
    python faster2dgs_eval.py -c configs/2DGS_m360.yaml ... --iters 7000 --profile-default
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Datasets as DI
    from Implementations import Methods as MI
    from Logging import Logger
    from faster2dgs_timings import (
        collect_run_timings,
        format_step_profile,
        format_train_timing,
        parse_timings_txt,
        profile_training_steps,
    )

# 2DGS paper Table 1 — MipNeRF360 test PSNR @ 30k
PAPER_PSNR_30K: dict[str, float] = {
    'bicycle': 24.87,
    'flowers': 21.95,
    'garden': 26.95,
    'stump': 26.47,
    'treehill': 22.49,
    'room': 31.75,
    'counter': 28.70,
    'kitchen': 30.50,
    'bonsai': 31.38,
}


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.is_file():
        return {}
    text = metrics_path.read_text()
    line = text.strip().splitlines()[-1] if text.strip() else ''
    return {key: float(val) for key, val in re.findall(r'(\w+):([\d.]+)', line)}


def _per_image_psnr(metrics_path: Path) -> list[tuple[int, float]]:
    if not metrics_path.is_file():
        return []
    rows: list[tuple[int, float]] = []
    for line in metrics_path.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append((int(parts[0]), float(parts[1])))
    return rows


def _scene_name_from_config() -> str:
    name = str(Framework.config.TRAINING.MODEL_NAME).lower()
    path = Path(str(Framework.config.DATASET.PATH)).name.lower()
    for scene in PAPER_PSNR_30K:
        if scene in name or scene in path:
            return scene
    return path or name


def _apply_overrides(overrides: list[str]) -> None:
    for item in overrides:
        key, value = item.split('=', 1)
        elements = key.split('.')
        target = Framework.config
        for part in elements[:-1]:
            target = getattr(target, part)
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        setattr(target, elements[-1], value)


def _configure_training_run(iters: int, *, enable_timing: bool = True) -> None:
    Framework.config.TRAINING.NUM_ITERATIONS = int(iters)
    Framework.config.TRAINING.GUI.ACTIVATE = False
    Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 1
    Framework.config.TRAINING.BACKUP.FINAL_CHECKPOINT = True
    Framework.config.TRAINING.BACKUP.RENDER_TESTSET = True
    Framework.config.TRAINING.BACKUP.INTERMEDIATE_RENDERINGS = True
    Framework.config.TRAINING.BACKUP.INTERVAL = int(iters)
    if enable_timing:
        Framework.config.TRAINING.TIMING.ACTIVATE = True


def _default_profile_iters(iters: int) -> list[int]:
    """End-of-run step timing (matches final splat count)."""
    return [max(0, iters - 1)] if iters > 0 else []


def train_and_eval(
    config_path: str,
    iters: int,
    overrides: list[str],
    *,
    profile_at: list[int] | None = None,
    profile_repeats: int = 15,
) -> Path:
    Framework.setup(config_path=config_path, require_custom_config=True)
    _apply_overrides(overrides)
    _configure_training_run(iters)
    scene = _scene_name_from_config()
    base_name = str(Framework.config.TRAINING.MODEL_NAME)
    if not base_name.endswith(f'_{iters}'):
        Framework.config.TRAINING.MODEL_NAME = f'{base_name}_eval{iters}'

    Logger.log_info(
        f'eval run: scene={scene}, iters={iters}, path={Framework.config.DATASET.PATH}, '
        f'model={Framework.config.TRAINING.MODEL_NAME}'
    )
    trainer = MI.get_training_instance(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=Framework.config.TRAINING.LOAD_CHECKPOINT,
    )
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer.run(dataset)
    run_dir = Path(trainer.output_directory)

    profile_iters = profile_at
    if profile_iters is None:
        profile_iters = []
    profiles = []
    if profile_iters:
        Logger.log_info(f'step profiling at iterations: {profile_iters}')
        profiles = profile_training_steps(
            trainer, dataset, profile_iters, repeats=profile_repeats,
        )

    collect_run_timings(run_dir, profile_steps=profiles or None)
    print_report(run_dir, iters, step_profiles=profiles)
    Framework.teardown()
    return run_dir


def render_only(run_dir: Path, checkpoint: str) -> Path:
    run_dir = run_dir.resolve()
    Framework.setup(config_path=str(run_dir / 'training_config.yaml'), require_custom_config=True)
    Framework.config.TRAINING.GUI.ACTIVATE = False
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(run_dir / 'checkpoints' / checkpoint),
    ).eval()
    renderer = MI.get_renderer(method=Framework.config.GLOBAL.METHOD_TYPE, model=model)
    dataset.test()
    renderer.render_subset(
        output_directory=run_dir,
        dataset=dataset,
        calculate_metrics=True,
        visualize_errors=False,
        verbose=True,
    )
    return run_dir


def _scene_from_run_dir(run_dir: Path) -> str:
    text = run_dir.name.lower()
    for scene in PAPER_PSNR_30K:
        if scene in text:
            return scene
    return text


def print_report(
    run_dir: Path,
    iters: int | None = None,
    *,
    step_profiles: list | None = None,
) -> dict[str, float]:
    run_dir = run_dir.resolve()
    if iters is None:
        test_dirs = sorted(run_dir.glob('test_*'), key=lambda p: int(p.name.split('_', 1)[1]))
        if test_dirs:
            iters = int(test_dirs[-1].name.split('_', 1)[1])
        else:
            iters = 0

    metrics_path = run_dir / f'test_{iters}' / 'metrics_8bit.txt'
    metrics = _parse_metrics(metrics_path)
    per_image = _per_image_psnr(metrics_path)
    n_gaussians_path = run_dir / 'n_gaussians.txt'
    n_gaussians = None
    if n_gaussians_path.is_file():
        for line in n_gaussians_path.read_text().splitlines():
            if line.startswith('N_Gaussians:'):
                n_gaussians = int(line.split(':', 1)[1].strip().replace(',', ''))

    scene = _scene_from_run_dir(run_dir)
    paper = PAPER_PSNR_30K.get(scene)
    psnr = metrics.get('PSNR')
    ssim = metrics.get('SSIM')
    lpips = metrics.get('LPIPS')

    print(f'\n=== Faster2DGS eval @ {iters:,} iters ===')
    print(f'  output: {run_dir}')
    if n_gaussians is not None:
        print(f'  splats: {n_gaussians:,}')
    if psnr is not None:
        print(f'  PSNR:  {psnr:.2f}  SSIM: {ssim:.3f}  LPIPS: {lpips:.3f}')
        if paper is not None:
            print(f'  paper @30k: {paper:.2f}  (delta {psnr - paper:+.2f} — not apples-to-apples if iters<{30000})')
    else:
        print(f'  metrics not found: {metrics_path}')

    if per_image:
        worst = sorted(per_image, key=lambda x: x[1])[:5]
        best = sorted(per_image, key=lambda x: x[1], reverse=True)[:3]
        print(f'  worst views: {", ".join(f"#{i}={v:.1f}" for i, v in worst)}')
        print(f'  best views:  {", ".join(f"#{i}={v:.1f}" for i, v in best)}')

    train = parse_timings_txt(run_dir / 'timings.txt')
    train_line = format_train_timing(train)
    if train_line:
        print(f'  {train_line}')

    profiles = step_profiles
    if profiles is None and (run_dir / 'timing_summary.json').is_file():
        import json
        from faster2dgs_timings import StepProfile

        data = json.loads((run_dir / 'timing_summary.json').read_text())
        profiles = [StepProfile(**row) for row in data.get('step_profiles', [])]
    if profiles:
        for p in profiles:
            print(f'  {format_step_profile(p)}')

    summary = run_dir / 'timing_summary.json'
    if summary.is_file() and (train_line or profiles):
        print(f'  timing_summary: {summary}')

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Faster2DGS train-to-N eval with test PSNR')
    parser.add_argument('-c', '--config', type=str, default=None, help='Training config yaml')
    parser.add_argument('--iters', type=int, default=10_000, help='Train iterations + test render (default 10000)')
    parser.add_argument('-d', '--run-dir', type=Path, default=None, help='Existing output dir (render-only mode)')
    parser.add_argument('--checkpoint', type=str, default='final.pt', help='Checkpoint for render-only')
    parser.add_argument('--render-only', action='store_true', help='Skip training; render test set from checkpoint')
    parser.add_argument(
        '--profile-at',
        type=int,
        nargs='*',
        default=None,
        help='Profile end-of-run CUDA step time after training (default: off)',
    )
    parser.add_argument(
        '--profile-default',
        action='store_true',
        help='Profile one training step at the final iteration (speed regression)',
    )
    parser.add_argument('--profile-repeats', type=int, default=15, help='Timed repeats per --profile-at iteration')
    parser.add_argument('overrides', nargs='*', help='Config overrides, e.g. DATASET.PATH=/path/to/scene')
    args, unknown = parser.parse_known_args()
    if unknown:
        args.overrides = list(args.overrides) + unknown

    if args.render_only:
        if args.run_dir is None:
            parser.error('--render-only requires -d/--run-dir')
        render_only(args.run_dir, args.checkpoint)
        Framework.setup(config_path=str(args.run_dir.resolve() / 'training_config.yaml'), require_custom_config=True)
        print_report(args.run_dir, args.iters)
        return

    if args.config is None:
        parser.error('training mode requires -c/--config')
    profile_at: list[int] | None = args.profile_at
    if args.profile_default:
        profile_at = _default_profile_iters(args.iters)
    train_and_eval(
        args.config,
        args.iters,
        args.overrides,
        profile_at=profile_at,
        profile_repeats=args.profile_repeats,
    )


if __name__ == '__main__':
    main()
