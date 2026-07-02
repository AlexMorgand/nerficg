#! /usr/bin/env python3
"""Run MipNeRF360 Faster2DGS photometric benchmark and compare PSNR to the 2DGS paper."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    from Logging import Logger

# 2DGS paper Table 1 (MipNeRF360 test PSNR)
PAPER_PSNR = {
    'bicycle': 24.87,
    'garden': 26.95,
    'stump': 26.47,
    'kitchen': 30.50,
}

# scene → (IMAGE_SCALE_FACTOR, outdoor|indoor)
M360_SCENES: dict[str, float] = {
    'bicycle': 0.25,
    'garden': 0.25,
    'stump': 0.25,
    'kitchen': 0.5,
}

CONFIG_REL = 'configs/2DGS_m360.yaml'


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.is_file():
        return {}
    text = metrics_path.read_text()
    line = text.strip().splitlines()[-1] if text.strip() else ''
    out: dict[str, float] = {}
    for key, val in re.findall(r'(\w+):([\d.]+)', line):
        out[key] = float(val)
    return out


def _find_latest_run(repo_root: Path, model_name: str) -> Path | None:
    base = repo_root / 'output' / 'Faster2DGS'
    if not base.is_dir():
        return None
    matches = sorted(base.glob(f'{model_name}_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_benchmark(scenes: list[str], output_root: Path | None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / CONFIG_REL
    if output_root is None:
        output_root = repo_root / 'output' / f'mip360_2dgs_benchmark_{datetime.now():%Y-%m-%d-%H-%M-%S}'
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float | None, float, str]] = []
    for scene in scenes:
        if scene not in M360_SCENES:
            Logger.log_warning(f'unknown scene {scene}, skipping')
            continue
        scale = M360_SCENES[scene]
        paper = PAPER_PSNR.get(scene, float('nan'))
        model_name = f'{scene}_m360_bench'
        dataset_path = f'dataset/mipnerf360/{scene}'
        Logger.log_info(f'=== training {scene} ({CONFIG_REL}, scale={scale}) ===')
        cmd = [
            sys.executable,
            str(repo_root / 'scripts' / 'train.py'),
            '-c', str(config_path),
            f'DATASET.PATH={dataset_path}',
            f'DATASET.IMAGE_SCALE_FACTOR={scale}',
            f'TRAINING.MODEL_NAME={model_name}',
            'TRAINING.GUI.ACTIVATE=False',
        ]
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
        except subprocess.CalledProcessError as exc:
            Logger.log_error(f'{scene} failed: {exc}')
            rows.append((scene, None, paper, str(exc)))
            continue

        run_dir = _find_latest_run(repo_root, model_name)
        if run_dir is None:
            rows.append((scene, None, paper, 'output directory not found'))
            continue
        metrics_path = run_dir / 'test_30000' / 'metrics_8bit.txt'
        metrics = _parse_metrics(metrics_path)
        psnr = metrics.get('PSNR')
        delta = (psnr - paper) if psnr is not None else float('nan')
        rows.append((scene, psnr, paper, str(run_dir)))
        if psnr is not None:
            Logger.log_info(f'{scene}: PSNR={psnr:.2f} (paper {paper:.2f}, delta {delta:+.2f})')
        else:
            Logger.log_warning(f'{scene}: no metrics at {metrics_path}')

    summary_path = output_root / 'mip360_2dgs_summary.txt'
    lines = ['scene\tours\tpaper\tdelta\toutput']
    for scene, ours, paper, note in rows:
        if ours is None:
            lines.append(f'{scene}\tFAILED\t{paper:.2f}\t\t{note}')
        else:
            lines.append(f'{scene}\t{ours:.2f}\t{paper:.2f}\t{ours - paper:+.2f}\t{note}')
    summary_path.write_text('\n'.join(lines) + '\n')
    Logger.log_info(f'Wrote summary: {summary_path}')
    print('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='MipNeRF360 Faster2DGS photometric paper comparison')
    parser.add_argument(
        '--scenes', nargs='*', default=list(M360_SCENES.keys()),
        help=f'Scenes to train (default: {list(M360_SCENES.keys())})',
    )
    parser.add_argument(
        '-o', '--output-root', type=Path, default=None,
        help='Directory for benchmark summary file',
    )
    args = parser.parse_args()
    run_benchmark(list(args.scenes), args.output_root)


if __name__ == '__main__':
    main()
