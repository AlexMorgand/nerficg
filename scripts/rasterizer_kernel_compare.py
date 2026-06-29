#!/usr/bin/env python3
"""Compare isolated CUDA kernel fwd+bwd: FasterGS 3D vs Faster2DGS surfel.

Run from ``scripts/``:

    python rasterizer_kernel_compare.py
    python rasterizer_kernel_compare.py --advance-to 3100 --method both
"""

from __future__ import annotations

import argparse
import time

import utils

with utils.DiscoverSourcePath():
    import torch
    import Framework
    from Implementations import Datasets as DI, Methods as MI
    from Methods.Faster2DGS.Faster2DGSCudaBackend import (
        SurfelRasterizerSettings,
        diff_rasterize_surfel_with_aux,
    )
    from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings, diff_rasterize
    from Methods.FasterGS.Renderer import extract_settings as extract_settings_3d


def _should_run_callback(callback, iteration: int) -> bool:
    if callback.start_iteration is not None and iteration < callback.start_iteration:
        return False
    if callback.end_iteration is not None and iteration > callback.end_iteration:
        return False
    if callback.iteration_stride is not None:
        base = callback.start_iteration or 0
        if (iteration - base) % callback.iteration_stride != 0:
            return False
    return True


def advance_trainer(trainer, dataset, iteration: int) -> None:
    for it in range(iteration):
        for cb in trainer._gather_callbacks(0):
            if _should_run_callback(cb, it):
                cb(trainer, it, dataset)


def setup(method: str, config_path: str):
    Framework.setup(config_path=config_path, require_custom_config=True)
    Framework.set_random_seed()
    Framework.config.GLOBAL.METHOD_TYPE = method
    Framework.config.TRAINING.GUI.ACTIVATE = False
    Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 1
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer = MI.get_training_instance(method, None)
    for cb in trainer._gather_callbacks(-1):
        cb(trainer, 0, dataset)
    return trainer, dataset


def time_kernel(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def bench_faster2dgs(trainer, dataset, warmup: int, repeats: int) -> tuple[int, float]:
    g = trainer.model.gaussians
    dataset.train()
    view = trainer.train_sampler.get(dataset=dataset)['view']
    bg = view.camera.background_color
    from Methods.FasterGS.Renderer import extract_settings
    settings = extract_settings(view, g.active_sh_bases, bg, trainer.renderer.PROPER_ANTIALIASING)
    dens = torch.zeros(2, g.means.shape[0], device='cuda')
    n = g.means.shape[0]

    def step():
        for p in (g._means, g._scales, g._rotations, g._opacities, g._sh_coefficients_0, g._sh_coefficients_rest):
            if p.grad is not None:
                p.grad = None
        rgb, allmap, _ = diff_rasterize_surfel_with_aux(
            means=g._means, raw_scales_2d=g._scales, rotations=g._rotations,
            opacities=g._opacities, sh_coefficients_0=g._sh_coefficients_0,
            sh_coefficients_rest=g._sh_coefficients_rest, densification_info=dens,
            rasterizer_settings=SurfelRasterizerSettings(*settings), view=view,
        )
        (rgb.mean() + 0.1 * allmap[6].mean() + 0.01 * allmap[:6].mean()).backward()

    ms = time_kernel(step, warmup, repeats)
    return n, ms


def bench_fastergs(trainer, dataset, warmup: int, repeats: int) -> tuple[int, float]:
    g = trainer.model.gaussians
    dataset.train()
    view = trainer.train_sampler.get(dataset=dataset)['view']
    bg = view.camera.background_color
    settings = extract_settings_3d(view, g.active_sh_bases, bg, trainer.renderer.PROPER_ANTIALIASING)
    dens = torch.zeros(2, g.means.shape[0], device='cuda')
    n = g.means.shape[0]

    def step():
        for p in (g._means, g._scales, g._rotations, g._opacities, g._sh_coefficients_0, g._sh_coefficients_rest):
            if p.grad is not None:
                p.grad = None
        rgb = diff_rasterize(
            means=g._means, scales=g._scales, rotations=g._rotations,
            opacities=g._opacities, sh_coefficients_0=g._sh_coefficients_0,
            sh_coefficients_rest=g._sh_coefficients_rest, densification_info=dens,
            rasterizer_settings=RasterizerSettings(*settings),
        )
        rgb.mean().backward()

    ms = time_kernel(step, warmup, repeats)
    return n, ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='../configs/gs_stump_2DGS.yaml')
    p.add_argument('--advance-to', type=int, default=0, help='Train callbacks before timing')
    p.add_argument('--method', choices=('both', 'Faster2DGS', 'FasterGS'), default='both')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    args = p.parse_args()

    results: list[tuple[str, int, float]] = []
    for method in (['Faster2DGS', 'FasterGS'] if args.method == 'both' else [args.method]):
        trainer, dataset = setup(method, args.config)
        if args.advance_to > 0:
            print(f'[{method}] advancing to iter {args.advance_to}...')
            advance_trainer(trainer, dataset, args.advance_to)
        if method == 'Faster2DGS':
            n, ms = bench_faster2dgs(trainer, dataset, args.warmup, args.repeats)
        else:
            n, ms = bench_fastergs(trainer, dataset, args.warmup, args.repeats)
        results.append((method, n, ms))
        print(f'[{method}] {n:,} splats  kernel fwd+bwd = {ms:.2f} ms  ({1000.0 / ms:.1f} steps/s)')
        Framework.teardown()

    if len(results) == 2:
        n0 = results[0][1]
        n1 = results[1][1]
        if n0 != n1:
            print(f'NOTE: splat counts differ ({n0:,} vs {n1:,}) — compare only if counts match.')
        ratio = results[0][2] / results[1][2]
        print(f'\nFaster2DGS / FasterGS kernel time ratio: {ratio:.2f}x')


if __name__ == '__main__':
    main()
