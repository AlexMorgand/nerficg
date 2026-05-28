#! /usr/bin/env python3

"""rank_metrics.py: Rank runs from metrics_8bit.txt files (overall + per metric)."""

from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


LOWER_IS_BETTER_HINTS = ('lpips', 'loss', 'error', 'rmse')


@dataclass
class RunMetrics:
    run_name: str
    metrics_path: Path
    metrics: dict[str, float]


def _parse_summary_metrics(metrics_file: Path) -> dict[str, float]:
    lines = metrics_file.read_text(encoding='utf-8').splitlines()
    for raw in reversed(lines):
        raw = raw.strip()
        if ':' not in raw:
            continue
        pairs = {}
        valid = True
        for token in raw.split():
            if ':' not in token:
                continue
            key, value = token.split(':', 1)
            try:
                pairs[key] = float(value)
            except ValueError:
                valid = False
                break
        if valid and pairs:
            return pairs
    raise ValueError(f'could not parse summary metrics from "{metrics_file}"')


def _discover_runs(root: Path, metrics_filename: str) -> list[RunMetrics]:
    metrics_files = sorted(root.rglob(metrics_filename))
    runs: list[RunMetrics] = []
    for metrics_file in metrics_files:
        try:
            metrics = _parse_summary_metrics(metrics_file)
        except Exception:
            continue
        run_name = str(metrics_file.parent.relative_to(root))
        runs.append(RunMetrics(run_name=run_name, metrics_path=metrics_file, metrics=metrics))
    return runs


def _metric_direction(metric_name: str) -> bool:
    """Returns True when higher is better."""
    name = metric_name.lower()
    return not any(hint in name for hint in LOWER_IS_BETTER_HINTS)


def _rank_rows(runs: list[RunMetrics], metric_name: str, higher_is_better: bool) -> list[tuple[int, RunMetrics, float]]:
    scored = [(run, run.metrics[metric_name]) for run in runs if metric_name in run.metrics]
    scored.sort(key=lambda item: item[1], reverse=higher_is_better)
    rows: list[tuple[int, RunMetrics, float]] = []
    for rank, (run, score) in enumerate(scored, start=1):
        rows.append((rank, run, score))
    return rows


def _compute_borda_overall(runs: list[RunMetrics], metric_names: Iterable[str]) -> list[tuple[int, RunMetrics, float]]:
    metric_names = [m for m in metric_names if all(m in run.metrics for run in runs)]
    if not metric_names:
        return []
    borda_scores = {id(run): 0.0 for run in runs}
    run_by_id = {id(run): run for run in runs}
    for metric_name in metric_names:
        higher_is_better = _metric_direction(metric_name)
        ranked = _rank_rows(runs, metric_name, higher_is_better)
        for rank, run, _ in ranked:
            borda_scores[id(run)] += rank
    rows = sorted(
        [(run_by_id[run_id], score / len(metric_names)) for run_id, score in borda_scores.items()],
        key=lambda item: item[1],
    )
    return [(rank, run, score) for rank, (run, score) in enumerate(rows, start=1)]


def _compute_mipnerf_overall(runs: list[RunMetrics]) -> list[tuple[int, RunMetrics, float]]:
    required = ('PSNR', 'SSIM', 'LPIPS')
    filtered = [run for run in runs if all(key in run.metrics for key in required)]
    if not filtered:
        return []
    scored = []
    for run in filtered:
        psnr = run.metrics['PSNR']
        ssim = run.metrics['SSIM']
        lpips = run.metrics['LPIPS']
        # Lower is better (same style as BaseTrainer._log_test_metrics_wandb).
        combined = math.exp(mean([
            -0.1 * math.log(10.0) * psnr,
            math.log(max(math.sqrt(max(1.0 - ssim, 1e-12)), 1e-12)),
            math.log(max(lpips, 1e-12)),
        ]))
        scored.append((run, combined))
    scored.sort(key=lambda item: item[1])
    return [(rank, run, score) for rank, (run, score) in enumerate(scored, start=1)]


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt='github')
    all_rows = [headers] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    def fmt(row: list[str]) -> str:
        return ' | '.join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
    sep = '-+-'.join('-' * w for w in widths)
    body = [fmt([str(cell) for cell in row]) for row in rows]
    return '\n'.join([fmt(headers), sep] + body)


def main() -> None:
    parser = ArgumentParser(
        prog='rank_metrics.py',
        description='Rank run outputs by parsing metrics_8bit.txt files.',
    )
    parser.add_argument(
        '-d', '--dir', dest='root', required=True,
        help='Root directory to recursively scan for metrics_8bit.txt files.'
    )
    parser.add_argument(
        '--metrics-file', dest='metrics_filename', default='metrics_8bit.txt',
        help='Metrics filename to scan for (default: metrics_8bit.txt).'
    )
    parser.add_argument(
        '--output', dest='output_filename', default='metrics_ranking.txt',
        help='Output ranking filename written under --dir.'
    )
    args, _ = parser.parse_known_args()

    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f'invalid --dir: "{root}"')

    runs = _discover_runs(root, args.metrics_filename)
    if not runs:
        raise RuntimeError(f'no "{args.metrics_filename}" files found under "{root}"')

    metric_names = sorted(set().union(*(run.metrics.keys() for run in runs)))
    sections: list[str] = []

    # Overall ranking by average rank across all common metrics.
    common_metrics = [m for m in metric_names if all(m in run.metrics for run in runs)]
    borda_rows = _compute_borda_overall(runs, common_metrics)
    if borda_rows:
        headers = ['Rank', 'Run', 'AvgRank'] + common_metrics + ['MetricsFile']
        table = []
        for rank, run, score in borda_rows:
            row = [rank, run.run_name, f'{score:.3f}']
            row.extend([f'{run.metrics[m]:.6f}' for m in common_metrics])
            row.append(run.metrics_path)
            table.append(row)
        sections.append(
            'OVERALL RANKING (Average Rank Across Common Metrics)\n'
            + f'Common metrics: {", ".join(common_metrics)}\n'
            + _format_table(table, headers=headers)
        )

    # Overall ranking using the mipnerf-style composite when possible.
    mip_rows = _compute_mipnerf_overall(runs)
    if mip_rows:
        headers = ['Rank', 'Run', 'Composite', 'PSNR', 'SSIM', 'LPIPS', 'MetricsFile']
        table = []
        for rank, run, score in mip_rows:
            table.append([
                rank,
                run.run_name,
                f'{score:.6f}',
                f'{run.metrics["PSNR"]:.6f}',
                f'{run.metrics["SSIM"]:.6f}',
                f'{run.metrics["LPIPS"]:.6f}',
                run.metrics_path,
            ])
        sections.append(
            'OVERALL RANKING (MipNeRF-Style Composite, lower is better)\n'
            + _format_table(table, headers=headers)
        )

    # Ranking per metric.
    for metric_name in metric_names:
        higher_is_better = _metric_direction(metric_name)
        ranked = _rank_rows(runs, metric_name, higher_is_better)
        table = [[rank, run.run_name, f'{value:.6f}', run.metrics_path] for rank, run, value in ranked]
        sections.append(
            f'PER-METRIC RANKING: {metric_name} '
            + ('(higher is better)' if higher_is_better else '(lower is better)')
            + '\n'
            + _format_table(table, headers=['Rank', 'Run', metric_name, 'MetricsFile'])
        )

    output_path = root / args.output_filename
    output_path.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    print(f'wrote ranking to {output_path}')


if __name__ == '__main__':
    main()
