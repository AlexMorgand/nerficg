#!/usr/bin/env python3
"""Benchmark Faster2DGS photometric fast paths vs full aux and project FastGS parity.

Compares kernel fwd+bwd at matched checkpoint iterations, with optional splat-count
scaling extrapolation toward official ~400k @3k.

Run from ``scripts/``:

    python faster2dgs_speed_experiment.py --profile-at 2000 3100
    python faster2dgs_speed_experiment.py --profile-only --profile-at 3100 --repeats 30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    import torch
    from Datasets.utils import apply_background_color, get_supervision_alpha
    from faster2dgs_profile_step import (
        CudaTimer,
        advance_to,
        setup_trainer,
        validate_dataset_path,
    )


OFFICIAL_2DGS_SPLAT_REF = 400_000


@dataclass
class KernelBench:
    splats: int
    photometric_only: bool
    fwd_ms: float
    bwd_ms: float
    total_ms: float

    @property
    def steps_per_s(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else float('inf')


def _prepare_step(trainer, dataset, iteration: int):
    trainer.model.train()
    dataset.train()
    trainer.loss.train()
    trainer.model.gaussians.update_learning_rate(iteration + 1)
    trainer.loss.set_geometry_loss_weights(
        distortion_scale=trainer._distortion_scale(iteration),
        normal_scale=trainer._normal_scale(iteration),
        depth_smoothness_scale=trainer._depth_smoothness_scale(iteration),
        planar_splat_scale=trainer._planar_splat_scale(iteration),
    )
    view = trainer.train_sampler.get(dataset=dataset)['view']
    bg = (
        torch.rand_like(view.camera.background_color)
        if trainer.USE_RANDOM_BACKGROUND_COLOR
        else view.camera.background_color
    )
    return view, bg


def _bench_rasterizer(
    trainer,
    dataset,
    iteration: int,
    *,
    photometric_only: bool,
    repeats: int,
    warmup: int,
) -> KernelBench:
    g = trainer.model.gaussians
    renderer = trainer.renderer
    update_dens = not trainer.USE_MCMC and iteration < trainer.DENSIFICATION_END_ITERATION
    splats = g.means.shape[0]

    fwd_acc = bwd_acc = 0.0
    n = warmup + repeats

    for i in range(n):
        view, bg = _prepare_step(trainer, dataset, iteration)
        rgb_gt = view.rgb
        if (alpha := get_supervision_alpha(view)) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha, bg)

        for p in (
            g._means, g._scales, g._rotations, g._opacities,
            g._sh_coefficients_0, g._sh_coefficients_rest,
        ):
            if p.grad is not None:
                p.grad = None

        with CudaTimer() as t_fwd:
            rgb, auxiliary_maps = renderer._rasterize_training(
                view, update_dens, bg, photometric_only=photometric_only,
            )
        if photometric_only:
            render_pkg = {'rgb': rgb}
        else:
            parsed = renderer._training_outputs_from_aux(view, auxiliary_maps)
            render_pkg = {'rgb': rgb, **parsed}

        with CudaTimer() as t_loss:
            loss = trainer.loss(render_pkg, rgb_gt)
        with CudaTimer() as t_bwd:
            loss.backward()

        if i >= warmup:
            fwd_acc += t_fwd.ms + t_loss.ms
            bwd_acc += t_bwd.ms

    fwd_ms = fwd_acc / repeats
    bwd_ms = bwd_acc / repeats
    return KernelBench(
        splats=splats,
        photometric_only=photometric_only,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        total_ms=fwd_ms + bwd_ms,
    )


def _print_bench(label: str, full: KernelBench, photo: KernelBench) -> None:
    speedup = full.total_ms / photo.total_ms if photo.total_ms > 0 else 0.0
    print(f'\n=== {label} ({full.splats:,} splats) ===')
    print(f'{"mode":<16} {"fwd+loss":>10} {"bwd":>10} {"total":>10} {"it/s":>8}')
    print('-' * 58)
    for b in (full, photo):
        tag = 'photometric' if b.photometric_only else 'full aux'
        print(
            f'{tag:<16} {b.fwd_ms:10.2f} {b.bwd_ms:10.2f} {b.total_ms:10.2f} {b.steps_per_s:8.1f}'
        )
    print(f'photometric speedup: {speedup:.2f}x total step')


def _project_fastgs_parity(photo: KernelBench, target_splats: int = OFFICIAL_2DGS_SPLAT_REF) -> None:
    """Linear splat scaling + fixed overhead heuristic toward FastGS-like targets."""
    scale = photo.splats / max(target_splats, 1)
    projected_ms = photo.total_ms / scale
    projected_its = 1000.0 / projected_ms if projected_ms > 0 else 0.0
    fastgs_kernel_ms = 1.5  # ~579k splats reference from kernel compare
    fastgs_its = 1000.0 / fastgs_kernel_ms
    print(f'\n=== projection @ {target_splats:,} splats (linear raster cost) ===')
    print(f'  current photometric: {photo.total_ms:.2f} ms @ {photo.splats:,} -> ~{projected_ms:.2f} ms ({projected_its:.0f} it/s)')
    print(f'  FastGS 3D kernel ref: ~{fastgs_kernel_ms:.1f} ms (~{fastgs_its:.0f} kernel steps/s)')
    print(f'  gap after splat budget: ~{projected_ms / fastgs_kernel_ms:.1f}x vs FastGS kernel-only')


def main() -> int:
    parser = argparse.ArgumentParser(description='Faster2DGS speed experiment')
    parser.add_argument('-c', '--config', default='../configs/gs_stump_2DGS.yaml')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--profile-at', type=int, nargs='+', default=[2000, 3100])
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--train-iters', type=int, default=0, help='Optional short train before bench')
    args = parser.parse_args()

    if args.dataset_path:
        validate_dataset_path(args.dataset_path)

    trainer, dataset = setup_trainer(args.config, args.dataset_path)

    if args.train_iters > 0 and not args.profile_only:
        from faster2dgs_profile_step import run_training_log
        run_training_log(trainer, dataset, args.train_iters)

    for iteration in args.profile_at:
        advance_to(trainer, dataset, iteration)
        needs_aux = trainer._needs_aux_maps(iteration)
        print(f'\niter {iteration}: {trainer.model.gaussians.means.shape[0]:,} splats, needs_aux={needs_aux}')

        full = _bench_rasterizer(
            trainer, dataset, iteration,
            photometric_only=False, repeats=args.repeats, warmup=args.warmup,
        )
        photo = _bench_rasterizer(
            trainer, dataset, iteration,
            photometric_only=True, repeats=args.repeats, warmup=args.warmup,
        )
        _print_bench(f'iter {iteration}', full, photo)
        if iteration < trainer.DISTORTION_START_ITERATION:
            _project_fastgs_parity(photo)

    return 0


if __name__ == '__main__':
    sys.exit(main())
