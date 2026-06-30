#!/usr/bin/env python3
"""7k photometric parity: Faster2DGS vs official 2DGS (official m360_eval defaults).

At iter 7000 both codebases match train.py gates with default lambda_dist=0:
  photometric only (L1+DSSIM); normal loss still off (iter > 7000).

Official m360_eval.py only differs by image folder (images_2 vs images_4).

Examples:
    python faster2dgs_parity_7k.py --scene kitchen
    python faster2dgs_parity_7k.py --scene stump --skip-official
    python faster2dgs_parity_7k.py --scene kitchen --skip-ours   # official only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    from faster2dgs_eval import PAPER_PSNR_30K, print_report, train_and_eval

OFFICIAL_PYTHON = Path('/opt/anaconda3/envs/pytorch_simulon_env/bin/python')
REPO = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO / 'dataset' / 'mipnerf360'
OFFICIAL_ROOT = Path.home() / 'Simulon' / 'dev' / '2d-gaussian-splatting'
ITERS = 7_000

SCENES: dict[str, dict] = {
    'stump': {'images': 'images_4', 'scale': 0.25, 'paper': 26.47},
    'garden': {'images': 'images_4', 'scale': 0.25, 'paper': 26.95},
    'bicycle': {'images': 'images_4', 'scale': 0.25, 'paper': 24.87},
    'kitchen': {'images': 'images_2', 'scale': 0.5, 'paper': 30.50},
    'room': {'images': 'images_2', 'scale': 0.5, 'paper': 31.75},
    'counter': {'images': 'images_2', 'scale': 0.5, 'paper': 28.70},
    'bonsai': {'images': 'images_2', 'scale': 0.5, 'paper': 31.38},
}


@dataclass
class RunResult:
    label: str
    run_dir: Path | None
    psnr: float | None
    ssim: float | None
    splats: int | None
    log_tail: str = ''


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.is_file():
        return {}
    line = metrics_path.read_text().strip().splitlines()[-1]
    return {k: float(v) for k, v in re.findall(r'(\w+):([\d.]+)', line)}


def _parse_splats(run_dir: Path) -> int | None:
    p = run_dir / 'n_gaussians.txt'
    if not p.is_file():
        return None
    for line in p.read_text().splitlines():
        if line.startswith('N_Gaussians:'):
            return int(line.split(':', 1)[1].strip().replace(',', ''))
    return None


def _collect_ours(scene: str, python: str) -> RunResult:
    meta = SCENES[scene]
    config = REPO / 'configs' / '2DGS_m360.yaml'
    path = DATASET_ROOT / scene
    overrides = [
        f'DATASET.PATH={path}',
        f'DATASET.IMAGE_SCALE_FACTOR={meta["scale"]}',
        f'TRAINING.MODEL_NAME={scene}_faster2dgs_parity7k',
    ]
    import os
    preload = os.environ.get('PRELOADING_LEVEL')
    if preload is not None:
        overrides.append(f'TRAINING.DATA.PRELOADING_LEVEL={preload}')
    run_dir = train_and_eval(str(config), ITERS, overrides)
    metrics = _parse_metrics(run_dir / f'test_{ITERS}' / 'metrics_8bit.txt')
    return RunResult(
        label='Faster2DGS',
        run_dir=run_dir,
        psnr=metrics.get('PSNR'),
        ssim=metrics.get('SSIM'),
        splats=_parse_splats(run_dir),
    )


def _official_results_json(out_dir: Path) -> dict | None:
    # metrics.py writes test/{iter}/results.json or scene_dir/results.json
    candidates = list(out_dir.rglob('results.json'))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def _collect_official(scene: str, python: str) -> RunResult:
    meta = SCENES[scene]
    if not OFFICIAL_ROOT.is_dir():
        return RunResult('official 2DGS', None, None, None, None, f'missing {OFFICIAL_ROOT}')
    out_dir = REPO / 'output' / 'official_2dgs' / f'{scene}_7k'
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(OFFICIAL_ROOT / 'train.py'),
        '-s', str(DATASET_ROOT / scene),
        '-i', meta['images'],
        '-m', str(out_dir),
        '--eval',
        '--iterations', str(ITERS),
        '--test_iterations', str(ITERS),
        '--save_iterations', str(ITERS),
        '--quiet',
    ]
    print(f'official cmd: {" ".join(cmd)}')
    proc = subprocess.run(cmd, cwd=str(OFFICIAL_ROOT), capture_output=True, text=True)
    log = (proc.stdout or '') + (proc.stderr or '')
    if proc.returncode != 0:
        return RunResult('official 2DGS', out_dir, None, None, None, log[-4000:])

    # Official --quiet training may skip printing test PSNR; render + metrics for parity.
    render_cmd = [
        python, str(OFFICIAL_ROOT / 'render.py'),
        '-m', str(out_dir), '--iteration', str(ITERS), '--skip_train', '--quiet',
    ]
    metrics_cmd = [python, str(OFFICIAL_ROOT / 'metrics.py'), '-m', str(out_dir)]
    subprocess.run(render_cmd, cwd=str(OFFICIAL_ROOT), capture_output=True, text=True)
    mproc = subprocess.run(metrics_cmd, cwd=str(OFFICIAL_ROOT), capture_output=True, text=True)
    log += (mproc.stdout or '') + (mproc.stderr or '')

    psnr = ssim = None
    # console: [ITER 7000] Evaluating test: L1 ... PSNR ...
    m = re.search(r'\[ITER\s+7000\].*Evaluating test:.*PSNR\s+([\d.]+)', log)
    if m:
        psnr = float(m.group(1))
    results = _official_results_json(out_dir)
    if results:
        key = f'ours_{ITERS}'
        if key in results:
            psnr = results[key].get('PSNR', psnr)
            ssim = results[key].get('SSIM', ssim)
    splats = None
    ply_stats = out_dir / 'point_cloud' / f'iteration_{ITERS}' / 'point_cloud.ply'
    if ply_stats.is_file():
        # vertex count from ply header
        with open(ply_stats, 'rb') as f:
            for line in f:
                s = line.decode('ascii', errors='ignore')
                if s.startswith('element vertex'):
                    splats = int(s.split()[-1])
                    break
    return RunResult('official 2DGS', out_dir, psnr, ssim, splats, log[-2000:])


def _print_table(scene: str, rows: list[RunResult]) -> None:
    paper = SCENES[scene]['paper']
    print(f'\n=== {scene} @ {ITERS} (photometric: λ_dist=0, normal off) ===')
    print(f'  paper reference @30k PSNR: {paper:.2f}')
    print(f'  {"method":<16} {"PSNR":>7} {"SSIM":>7} {"splats":>10}  output')
    for r in rows:
        psnr = f'{r.psnr:.2f}' if r.psnr is not None else 'n/a'
        ssim = f'{r.ssim:.3f}' if r.ssim is not None else 'n/a'
        splats = f'{r.splats:,}' if r.splats is not None else 'n/a'
        out = str(r.run_dir) if r.run_dir else '-'
        print(f'  {r.label:<16} {psnr:>7} {ssim:>7} {splats:>10}  {out}')
        if r.psnr is not None:
            print(f'    vs paper@30k: {r.psnr - paper:+.2f} dB')

    ours = next((r for r in rows if r.label == 'Faster2DGS'), None)
    off = next((r for r in rows if r.label == 'official 2DGS'), None)
    if ours and off and ours.psnr is not None and off.psnr is not None:
        delta = ours.psnr - off.psnr
        ratio = f', splats ratio {ours.splats / off.splats:.2f}x' if ours.splats and off.splats else ''
        print(f'  Faster2DGS vs official @7k: PSNR {delta:+.2f} dB{ratio}')

    for r in rows:
        if r.log_tail and r.psnr is None:
            print(f'\n--- {r.label} log tail ---\n{r.log_tail}')


def main() -> None:
    parser = argparse.ArgumentParser(description='7k pre-normal Faster2DGS vs official 2DGS')
    parser.add_argument('--scene', choices=sorted(SCENES), required=True)
    parser.add_argument('--skip-ours', action='store_true')
    parser.add_argument('--skip-official', action='store_true')
    parser.add_argument('--python', default=sys.executable, help='Python for Faster2DGS')
    parser.add_argument('--official-python', default=str(OFFICIAL_PYTHON), help='Python for official 2DGS')
    args = parser.parse_args()

    rows: list[RunResult] = []
    if not args.skip_ours:
        print(f'--- training Faster2DGS {args.scene} @ {ITERS} ---')
        rows.append(_collect_ours(args.scene, args.python))
    if not args.skip_official:
        print(f'--- training official 2DGS {args.scene} @ {ITERS} ---')
        rows.append(_collect_official(args.scene, args.official_python))
    _print_table(args.scene, rows)


if __name__ == '__main__':
    main()
