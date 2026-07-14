#! /usr/bin/env python3
"""Train FasterGS-DR on 3DGS-DR benchmark scenes and summarize test metrics."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'configs'

# Table 1 targets ("Ours, deferred") for quick comparison after a run.
TARGET_METRICS: dict[str, dict[str, float]] = {
    'ball': {'PSNR': 33.66, 'SSIM': 0.979, 'LPIPS': 0.098},
    'car': {'PSNR': 30.39, 'SSIM': 0.962, 'LPIPS': 0.033},
    'coffee': {'PSNR': 34.65, 'SSIM': 0.976, 'LPIPS': 0.076},
    'helmet': {'PSNR': 31.69, 'SSIM': 0.971, 'LPIPS': 0.049},
    'teapot': {'PSNR': 47.12, 'SSIM': 0.997, 'LPIPS': 0.005},
    'toaster': {'PSNR': 27.02, 'SSIM': 0.943, 'LPIPS': 0.081},
    'bell': {'PSNR': 31.65, 'SSIM': 0.962, 'LPIPS': 0.046},
    'cat': {'PSNR': 33.86, 'SSIM': 0.976, 'LPIPS': 0.040},
    'luyu': {'PSNR': 28.71, 'SSIM': 0.936, 'LPIPS': 0.053},
    'potion': {'PSNR': 32.29, 'SSIM': 0.957, 'LPIPS': 0.075},
    'tbell': {'PSNR': 28.94, 'SSIM': 0.952, 'LPIPS': 0.067},
    'teapot_glossy': {'PSNR': 25.36, 'SSIM': 0.936, 'LPIPS': 0.067},
}

DEFAULT_SCENES = tuple(TARGET_METRICS.keys())
METRIC_LINE = re.compile(r'^(PSNR|SSIM|LPIPS)\s+([\d.]+)\s*$', re.MULTILINE)


@dataclass(frozen=True)
class RunResult:
    scene: str
    output_dir: Path | None
    metrics: dict[str, float]
    error: str | None = None


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    text = metrics_path.read_text(encoding='utf-8')
    parsed: dict[str, float] = {}
    for match in METRIC_LINE.finditer(text):
        parsed[match.group(1)] = float(match.group(2))
    return parsed


def _find_metrics(output_root: Path, model_name: str) -> Path | None:
    candidates = sorted(output_root.glob(f'**/{model_name}_*/**/metrics_8bit.txt'))
    if not candidates:
        candidates = sorted(output_root.glob(f'**/{model_name}_*/metrics_8bit.txt'))
    return candidates[-1] if candidates else None


def _run_scene(
    scene: str,
    python: str,
    dry_run: bool,
    extra_args: list[str],
) -> RunResult:
    config_path = CONFIG_DIR / f'fastergs_dr_{scene}.yaml'
    if not config_path.is_file():
        return RunResult(scene=scene, output_dir=None, metrics={}, error=f'missing config {config_path}')

    cmd = [python, str(REPO_ROOT / 'scripts' / 'train.py'), '-c', str(config_path), *extra_args]
    print(f'\n=== {scene} ===')
    print(' '.join(cmd))
    if dry_run:
        return RunResult(scene=scene, output_dir=None, metrics={})

    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        return RunResult(scene=scene, output_dir=None, metrics={}, error=f'training exit {proc.returncode}')

    metrics_path = _find_metrics(REPO_ROOT / 'output' / 'FasterGS', f'fastergs_dr_{scene}')
    if metrics_path is None:
        return RunResult(scene=scene, output_dir=None, metrics={}, error='metrics_8bit.txt not found')
    return RunResult(
        scene=scene,
        output_dir=metrics_path.parent,
        metrics=_parse_metrics(metrics_path),
    )


def _print_summary(results: list[RunResult]) -> None:
    header = f'{"Scene":<14} {"PSNR":>8} {"tgt":>8} {"SSIM":>8} {"tgt":>8} {"LPIPS":>8} {"tgt":>8}  note'
    print('\n' + header)
    print('-' * len(header))
    for result in results:
        tgt = TARGET_METRICS.get(result.scene, {})
        if result.error:
            print(f'{result.scene:<14} {"—":>8} {"—":>8} {"—":>8} {"—":>8} {"—":>8} {"—":>8}  {result.error}')
            continue
        psnr = result.metrics.get('PSNR')
        ssim = result.metrics.get('SSIM')
        lpips = result.metrics.get('LPIPS')
        print(
            f'{result.scene:<14} '
            f'{psnr:8.2f} {tgt.get("PSNR", 0):8.2f} '
            f'{ssim:8.3f} {tgt.get("SSIM", 0):8.3f} '
            f'{lpips:8.3f} {tgt.get("LPIPS", 0):8.3f}  '
            f'{result.output_dir}'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description='Run FasterGS-DR benchmark scenes (3DGS-DR Table 1 parity).')
    parser.add_argument('--scenes', nargs='*', default=list(DEFAULT_SCENES), help='Scene names (default: all Table-1 scenes).')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter with nerficg/CUDA deps.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training.')
    parser.add_argument('--generate-configs', action='store_true', help='Regenerate benchmark YAMLs before training.')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, help='Extra args forwarded to train.py after "--".')
    args = parser.parse_args()

    if args.generate_configs:
        subprocess.run([sys.executable, str(REPO_ROOT / 'scripts' / 'generate_fastergs_dr_benchmark_configs.py')], check=True)

    extra = args.extra
    if extra and extra[0] == '--':
        extra = extra[1:]

    results = [_run_scene(scene, args.python, args.dry_run, extra) for scene in args.scenes]
    if not args.dry_run:
        _print_summary(results)
    return 0 if all(r.error is None for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
