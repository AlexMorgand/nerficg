#!/usr/bin/env python3
"""Log densify clone/split/prune breakdown vs official 2DGS reference milestones.

Run from ``scripts/``:

    python faster2dgs_densify_compare.py -c ../configs/2DGS_m360.yaml --iters 3500
    python faster2dgs_densify_compare.py --iters 7000 --milestones 500,1000,1700,3000,3500,7000
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    from faster2dgs_profile_step import (
        collect_densify_record,
        is_densify_iteration,
        run_callbacks,
        setup_trainer,
        validate_dataset_path,
    )


# Reported / measured official 2DGS splat counts (MipNeRF360 stump, ~0.25 res).
OFFICIAL_2DGS_MILESTONES: dict[int, int] = {
    500: 40_000,
    1000: 120_000,
    1700: 400_000,
    3000: 400_000,
    7000: 450_000,
    15000: 500_000,
}


@dataclass
class DensifyRow:
    iteration: int
    n_before: int
    n_after: int
    grad_candidates: int
    n_duplicate: int
    n_split_parents: int
    n_new: int
    n_pruned: int
    opacity_pruned: int
    screen_pruned: int
    world_pruned: int
    split_pruned: int

    @property
    def net(self) -> int:
        return self.n_after - self.n_before

    @property
    def grad_rate(self) -> float:
        return self.grad_candidates / max(self.n_before, 1)

    @property
    def prune_rate(self) -> float:
        return self.n_pruned / max(self.n_before, 1)


def _collect_row(trainer, iteration: int) -> DensifyRow | None:
    g = trainer.model.gaussians
    stats = getattr(g, '_last_densify_stats', None)
    breakdown = getattr(g, '_last_prune_breakdown', None)
    if stats is None:
        return None
    return DensifyRow(
        iteration=iteration,
        n_before=int(stats['n_before']),
        n_after=int(stats['n_after']),
        grad_candidates=int(stats.get('grad_candidates', 0)),
        n_duplicate=int(stats['n_duplicate']),
        n_split_parents=int(stats['n_split_parents']),
        n_new=int(stats['n_new']),
        n_pruned=int(stats['n_pruned']),
        opacity_pruned=int(breakdown['opacity']) if breakdown else 0,
        screen_pruned=int(breakdown['screen_size']) if breakdown else 0,
        world_pruned=int(breakdown['world_scale']) if breakdown else 0,
        split_pruned=int(breakdown['split_parents']) if breakdown else 0,
    )


def run_compare(config_path: str, dataset_path: str | None, num_iters: int, milestones: set[int]) -> list[DensifyRow]:
    trainer, dataset = setup_trainer(config_path, dataset_path)
    rows: list[DensifyRow] = []
    g = trainer.model.gaussians

    print(f'densify compare: {num_iters} iters, init={g.means.shape[0]:,}')
    print(f'  grad_threshold={trainer.DENSIFICATION_GRAD_THRESHOLD}')
    print(f'  opacity_cull={trainer.DENSIFICATION_OPACITY_CULL}, reset_max={trainer.OPACITY_RESET_MAX}')
    print(f'  densify {trainer.DENSIFICATION_START_ITERATION}..{trainer.DENSIFICATION_END_ITERATION} '
          f'every {trainer.DENSIFICATION_INTERVAL}')
    print()

    for iteration in range(num_iters):
        run_callbacks(trainer, dataset, iteration)
        if is_densify_iteration(trainer, iteration):
            row = _collect_row(trainer, iteration)
            if row is not None:
                rows.append(row)
        if iteration in milestones and not is_densify_iteration(trainer, iteration):
            print(f'  iter {iteration:5d} splats={g.means.shape[0]:,} (mid-interval)')

    return rows


def print_table(rows: list[DensifyRow], milestones: set[int]) -> None:
    header = (
        f'{"iter":>6} {"before":>10} {"after":>10} {"net":>8} '
        f'{"grad%":>6} {"dup":>7} {"split":>7} {"new":>7} {"pruned":>8} '
        f'{"op_pr":>7} {"scr_pr":>7} {"official":>10} {"ratio":>6}'
    )
    print(header)
    print('-' * len(header))

    by_iter = {r.iteration: r for r in rows}
    check_iters = sorted(set(by_iter) | milestones)

    for it in check_iters:
        if it not in by_iter:
            continue
        r = by_iter[it]
        official = OFFICIAL_2DGS_MILESTONES.get(it)
        ratio = f'{r.n_after / official:.2f}x' if official else '—'
        off_str = f'{official:,}' if official else '—'
        print(
            f'{r.iteration:6d} {r.n_before:10,} {r.n_after:10,} {r.net:+8,} '
            f'{100 * r.grad_rate:5.1f}% {r.n_duplicate:7,} {r.n_split_parents:7,} {r.n_new:7,} {r.n_pruned:8,} '
            f'{r.opacity_pruned:7,} {r.screen_pruned:7,} {off_str:>10} {ratio:>6}'
        )

    if rows:
        last = rows[-1]
        print()
        print('Summary (last densify):')
        print(f'  grad trigger rate: {100 * last.grad_rate:.1f}% of splats pass threshold')
        print(f'  yield per densify: +{last.n_new:,} new, -{last.n_pruned:,} pruned ({last.net:+,} net)')
        if last.n_pruned > 0:
            print(
                f'  prune mix: split_parents={last.split_pruned:,}, opacity={last.opacity_pruned:,}, '
                f'screen={last.screen_pruned:,}, world={last.world_pruned:,}'
            )


def write_csv(rows: list[DensifyRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'iteration', 'n_before', 'n_after', 'net', 'grad_candidates', 'grad_rate',
            'n_duplicate', 'n_split_parents', 'n_new', 'n_pruned',
            'opacity_pruned', 'screen_pruned', 'world_pruned', 'split_pruned',
            'official_ref', 'vs_official',
        ])
        for r in rows:
            official = OFFICIAL_2DGS_MILESTONES.get(r.iteration)
            w.writerow([
                r.iteration, r.n_before, r.n_after, r.net, r.grad_candidates, f'{r.grad_rate:.4f}',
                r.n_duplicate, r.n_split_parents, r.n_new, r.n_pruned,
                r.opacity_pruned, r.screen_pruned, r.world_pruned, r.split_pruned,
                official or '', f'{r.n_after / official:.3f}' if official else '',
            ])
    print(f'\nWrote {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Faster2DGS densify parity log')
    parser.add_argument('-c', '--config', default='../configs/2DGS_m360.yaml')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--iters', type=int, default=3500)
    parser.add_argument('--milestones', default='500,1000,1700,3000,3500,7000')
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    if args.dataset_path:
        validate_dataset_path(args.dataset_path)

    milestones = {int(x) for x in args.milestones.split(',')}
    rows = run_compare(args.config, args.dataset_path, args.iters, milestones)
    print()
    print_table(rows, milestones)
    if args.csv:
        write_csv(rows, Path(args.csv))
    return 0


if __name__ == '__main__':
    sys.exit(main())
