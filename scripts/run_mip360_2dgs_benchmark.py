#! /usr/bin/env python3
"""Run MipNeRF360 Faster2DGS benchmark configs and compare PSNR to the 2DGS paper."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    import Framework
    import train
    from Logging import Logger

# 2DGS paper Table 1 (MipNeRF360 test PSNR)
PAPER_PSNR = {
    'bicycle': 24.87,
    'garden': 26.95,
    'stump': 26.47,
    'kitchen': 30.50,
}

DEFAULT_CONFIGS = [
    'configs/gs_bicycle_2DGS.yaml',
    'configs/gs_stump_2DGS.yaml',
    'configs/gs_kitchen_2DGS.yaml',
    'configs/gs_garden_2DGS.yaml',
]


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.is_file():
        return {}
    text = metrics_path.read_text()
    line = text.strip().splitlines()[-1] if text.strip() else ''
    out: dict[str, float] = {}
    for key, val in re.findall(r'(\w+):([\d.]+)', line):
        out[key] = float(val)
    return out


def _scene_name(config_path: Path) -> str:
    stem = config_path.stem
    for scene in PAPER_PSNR:
        if scene in stem:
            return scene
    return stem


def run_benchmark(config_paths: list[Path], output_root: Path | None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if output_root is None:
        output_root = repo_root / 'output' / f'mip360_2dgs_benchmark_{datetime.now():%Y-%m-%d-%H-%M-%S}'
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float | None, float, str]] = []
    for config_path in config_paths:
        config_path = config_path if config_path.is_absolute() else repo_root / config_path
        scene = _scene_name(config_path)
        paper = PAPER_PSNR.get(scene, float('nan'))
        Logger.log_info(f'=== training {scene} ({config_path.name}) ===')
        try:
            training_instance = train.main(config_path=str(config_path))
        except Exception as exc:
            Logger.log_error(f'{scene} failed: {exc}')
            rows.append((scene, None, paper, str(exc)))
            continue

        run_dir = Path(training_instance.output_directory)
        metrics_path = run_dir / f'test_{training_instance.NUM_ITERATIONS}' / 'metrics_8bit.txt'
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
    parser = argparse.ArgumentParser(description='MipNeRF360 Faster2DGS paper comparison')
    parser.add_argument(
        '-c', '--configs', nargs='*', default=DEFAULT_CONFIGS,
        help='Training config paths (default: four gs_*_2DGS.yaml scenes)',
    )
    parser.add_argument(
        '-o', '--output-root', type=Path, default=None,
        help='Directory for benchmark summary file',
    )
    args = parser.parse_args()
    run_benchmark([Path(p) for p in args.configs], args.output_root)


if __name__ == '__main__':
    main()
