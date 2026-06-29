#!/usr/bin/env python3
"""Profile Faster2DGS training: per-component CUDA timing + splat count vs iteration.

Run from ``scripts/``:

    python faster2dgs_profile_step.py
    python faster2dgs_profile_step.py -c ../configs/gs_stump_2DGS.yaml --iters 3500 --profile-at 3100
    python faster2dgs_profile_step.py --profile-only --profile-at 1500 --repeats 50 --kernel-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import utils

with utils.DiscoverSourcePath():
    import torch
    import Framework
    from Datasets.utils import apply_background_color, get_supervision_alpha
    from Implementations import Datasets as DI, Methods as MI
    from Methods.Faster2DGS.Faster2DGSCudaBackend import SurfelRasterizerSettings, diff_rasterize_surfel_with_aux
    from Methods.FasterGS.Renderer import extract_settings


OFFICIAL_2DGS_SPLAT_REF = 400_000


@dataclass
class DensifyRecord:
    iteration: int
    n_before: int
    n_after: int
    n_pruned: int
    n_new: int
    opacity_pruned: int | None = None
    screen_pruned: int | None = None


@dataclass
class StepTimings:
    labels: list[str]
    ms: list[float]

    def total_ms(self) -> float:
        return sum(self.ms)

    def print_table(self, splats: int, resolution: str) -> None:
        total = self.total_ms()
        print(f'\n=== step breakdown @ {splats:,} splats, {resolution} ===')
        print(f'{"component":<22} {"ms":>8} {"%":>6} {"steps/s":>8}')
        print('-' * 48)
        for label, t in zip(self.labels, self.ms):
            pct = 100.0 * t / total if total > 0 else 0.0
            rate = 1000.0 / t if t > 0 else float('inf')
            print(f'{label:<22} {t:8.2f} {pct:5.1f}% {rate:8.1f}')
        print('-' * 48)
        print(f'{"TOTAL":<22} {total:8.2f} {"100.0":>5}% {1000.0 / total if total > 0 else 0.0:8.1f}')


class CudaTimer:
    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.ms = 0.0

    def __enter__(self) -> 'CudaTimer':
        self._start.record()
        return self

    def __exit__(self, *_args) -> None:
        self._end.record()
        torch.cuda.synchronize()
        self.ms = self._start.elapsed_time(self._end)


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


def validate_dataset_path(path: str) -> Path:
    root = Path(path).resolve()
    sparse = root / 'sparse' / '0'
    if not sparse.is_dir():
        raise FileNotFoundError(
            f'COLMAP sparse/0 not found at {sparse}\n'
            f'  Stump/Mip360 data is symlinked via dataset/mipnerf360 -> '
            f'/home/alex/Simulon/data/Misc/MipNerf/360_v2/\n'
            f'  which requires the external drive at /media/alex/morgandcorp_data_center to be mounted.\n'
            f'  Mount the drive, or pass --dataset-path to a local COLMAP scene.'
        )
    return root


def setup_trainer(config_path: str, dataset_path: str | None = None):
    Framework.setup(config_path=config_path, require_custom_config=True)
    Framework.set_random_seed()
    if dataset_path is not None:
        Framework.config.DATASET.PATH = str(validate_dataset_path(dataset_path))
    else:
        validate_dataset_path(Framework.config.DATASET.PATH)
    Framework.config.TRAINING.GUI.ACTIVATE = False
    Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 1
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer = MI.get_training_instance(Framework.config.GLOBAL.METHOD_TYPE, None)
    for cb in trainer._gather_callbacks(-1):
        cb(trainer, 0, dataset)
    return trainer, dataset


def run_callbacks(trainer, dataset, iteration: int) -> None:
    for cb in trainer._gather_callbacks(0):
        if not _should_run_callback(cb, iteration):
            continue
        cb(trainer, iteration, dataset)


def advance_to(trainer, dataset, iteration: int) -> None:
    for it in range(iteration):
        run_callbacks(trainer, dataset, it)


def collect_densify_record(trainer, iteration: int) -> DensifyRecord | None:
    g = trainer.model.gaussians
    stats = getattr(g, '_last_densify_stats', None)
    if stats is None:
        return None
    breakdown = getattr(g, '_last_prune_breakdown', None)
    return DensifyRecord(
        iteration=iteration,
        n_before=int(stats['n_before']),
        n_after=int(stats['n_after']),
        n_pruned=int(stats['n_pruned']),
        n_new=int(stats['n_new']),
        opacity_pruned=int(breakdown['opacity']) if breakdown else None,
        screen_pruned=int(breakdown['screen_size']) if breakdown else None,
    )


def is_densify_iteration(trainer, iteration: int) -> bool:
    if trainer.USE_MCMC:
        return False
    start = trainer.DENSIFICATION_START_ITERATION
    end = trainer.DENSIFICATION_END_ITERATION
    stride = trainer.DENSIFICATION_INTERVAL
    return start <= iteration <= end and (iteration - start) % stride == 0


def run_training_log(trainer, dataset, num_iters: int) -> list[DensifyRecord]:
    records: list[DensifyRecord] = []
    g = trainer.model.gaussians
    n_init = g.means.shape[0]
    milestones = {500, 1000, 1500, 2000, 2500, 3000, 3100, 3500, 7000, 15000}
    print(f'training log: {num_iters} iters, init splats={n_init:,}')
    t0 = time.perf_counter()

    for iteration in range(num_iters):
        run_callbacks(trainer, dataset, iteration)
        if is_densify_iteration(trainer, iteration):
            rec = collect_densify_record(trainer, iteration)
            if rec is not None:
                records.append(rec)
                extra = ''
                if rec.opacity_pruned is not None:
                    extra = f' [opacity={rec.opacity_pruned:,}, screen={rec.screen_pruned:,}]'
                print(
                    f'  iter {rec.iteration:5d} densify: {rec.n_before:,} -> {rec.n_after:,} '
                    f'(+{rec.n_new:,}, -{rec.n_pruned:,}){extra}'
                )
        elif iteration in milestones:
            print(f'  iter {iteration:5d} splats={g.means.shape[0]:,}')

    elapsed = time.perf_counter() - t0
    n_final = g.means.shape[0]
    print(
        f'training log done: {n_final:,} splats ({n_final - n_init:+,}), '
        f'{elapsed:.1f}s ({num_iters / elapsed:.1f} it/s avg)'
    )
    return records


def _prepare_view(trainer, dataset):
    trainer.model.train()
    dataset.train()
    trainer.loss.train()
    view = trainer.train_sampler.get(dataset=dataset)['view']
    bg_color = (
        torch.rand_like(view.camera.background_color)
        if trainer.USE_RANDOM_BACKGROUND_COLOR
        else view.camera.background_color
    )
    return view, bg_color


def _set_iter_schedules(trainer, iteration: int) -> None:
    trainer.model.gaussians.update_learning_rate(iteration + 1)
    trainer.loss.set_geometry_loss_weights(
        distortion_scale=trainer._distortion_scale(iteration),
        normal_scale=trainer._warmup_scale(iteration, trainer.NORMAL_START_ITERATION),
        depth_smoothness_scale=trainer._warmup_scale(iteration, trainer.DEPTH_SMOOTHNESS_START_ITERATION),
        planar_splat_scale=trainer._warmup_scale(iteration, trainer.PLANAR_SPLAT_START_ITERATION),
    )


def profile_step(trainer, dataset, iteration: int, repeats: int, warmup: int) -> StepTimings:
    """CUDA-event breakdown of one training step at ``iteration``."""
    g = trainer.model.gaussians
    advance_to(trainer, dataset, iteration)
    _set_iter_schedules(trainer, iteration)

    splats = g.means.shape[0]
    view, _bg = _prepare_view(trainer, dataset)
    w, h = view.camera.width, view.camera.height
    update_dens = not trainer.USE_MCMC and iteration < trainer.DENSIFICATION_END_ITERATION

    print(f'profiling step @ iter {iteration} ({splats:,} splats, {w}x{h})...')
    accum: dict[str, float] = {}

    def _accum(label: str, dt: float) -> None:
        accum[label] = accum.get(label, 0.0) + dt

    def _one_step() -> None:
        for p in (
            g._means, g._scales, g._rotations, g._opacities,
            g._sh_coefficients_0, g._sh_coefficients_rest,
        ):
            if p.grad is not None:
                p.grad = None

        with CudaTimer() as t_sample:
            view_i, bg_i = _prepare_view(trainer, dataset)
        _accum('sample_view', t_sample.ms)

        renderer = trainer.renderer
        photometric_only = not trainer._needs_aux_maps(iteration)
        with CudaTimer() as t_rast:
            rgb, auxiliary_maps = renderer._rasterize_training(
                view_i, update_dens, bg_i, photometric_only=photometric_only,
            )
        _accum('rasterize_fwd', t_rast.ms)

        if photometric_only:
            render_pkg = {'rgb': rgb}
            if update_dens and hasattr(renderer, '_last_training_radii'):
                render_pkg['radii'] = renderer._last_training_radii
        else:
            with CudaTimer() as t_aux:
                parsed = renderer._training_outputs_from_aux(view_i, auxiliary_maps)
                render_pkg = {'rgb': rgb, **parsed}
                if update_dens and hasattr(renderer, '_last_training_radii'):
                    render_pkg['radii'] = renderer._last_training_radii
            _accum('aux_parse', t_aux.ms)

        rgb_gt = view_i.rgb
        if (supervision_alpha := get_supervision_alpha(view_i)) is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_i)

        with CudaTimer() as t_loss:
            loss = trainer.loss(render_pkg, rgb_gt)
        _accum('loss_fwd', t_loss.ms)

        with CudaTimer() as t_bwd:
            loss.backward()
        _accum('backward', t_bwd.ms)

        with CudaTimer() as t_opt:
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
            g.post_optimizer_step(inject_noise=trainer.USE_MCMC)
        _accum('optimizer', t_opt.ms)

        if (
            update_dens
            and (radii := render_pkg.get('radii')) is not None
            and radii.numel() > 0
            and hasattr(g, 'update_max_radii2D')
        ):
            with CudaTimer() as t_rad:
                with torch.no_grad():
                    g.update_max_radii2D(radii)
            _accum('max_radii2d', t_rad.ms)

    for _ in range(warmup):
        _one_step()
    accum.clear()

    for _ in range(repeats):
        _one_step()

    order = ['sample_view', 'rasterize_fwd', 'aux_parse', 'loss_fwd', 'backward', 'optimizer', 'max_radii2d']
    labels, ms = [], []
    for key in order:
        if key in accum:
            labels.append(key)
            ms.append(accum[key] / repeats)

    timings = StepTimings(labels=labels, ms=ms)
    timings.print_table(splats, f'{w}x{h}')
    return timings


def profile_kernel_only(trainer, dataset, iteration: int, repeats: int, warmup: int) -> float:
    """Isolated native surfel fwd+bwd (no aux parse / Python loss)."""
    g = trainer.model.gaussians
    advance_to(trainer, dataset, iteration)
    view, bg_color = _prepare_view(trainer, dataset)
    settings = extract_settings(view, g.active_sh_bases, bg_color, trainer.renderer.PROPER_ANTIALIASING)
    dens = torch.zeros(2, g.means.shape[0], device='cuda')
    splats = g.means.shape[0]

    def step() -> None:
        for p in (g._means, g._scales, g._rotations, g._opacities, g._sh_coefficients_0, g._sh_coefficients_rest):
            if p.grad is not None:
                p.grad = None
        rgb, allmap, _radii = diff_rasterize_surfel_with_aux(
            means=g._means, raw_scales_2d=g._scales, rotations=g._rotations,
            opacities=g._opacities, sh_coefficients_0=g._sh_coefficients_0,
            sh_coefficients_rest=g._sh_coefficients_rest, densification_info=dens,
            rasterizer_settings=SurfelRasterizerSettings(*settings), view=view,
        )
        (rgb.mean() + 0.1 * allmap[6].mean() + 0.01 * allmap[:6].mean()).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1000.0
    print(f'\n=== isolated kernel fwd+bwd @ {splats:,} splats === {ms:.2f} ms ({1000.0 / ms:.1f} steps/s)')
    return ms


def write_csv(path: Path, records: list[DensifyRecord], splats_now: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'n_before', 'n_after', 'n_new', 'n_pruned', 'opacity_pruned', 'screen_pruned'])
        for r in records:
            w.writerow([
                r.iteration, r.n_before, r.n_after, r.n_new, r.n_pruned,
                r.opacity_pruned if r.opacity_pruned is not None else '',
                r.screen_pruned if r.screen_pruned is not None else '',
            ])
        w.writerow(['final', '', splats_now, '', '', '', ''])
    print(f'wrote {path}')


def print_splat_summary(records: list[DensifyRecord], splats_now: int) -> None:
    print('\n=== splat count summary ===')
    print(f'  current:               {splats_now:,}')
    print(f'  official 2DGS ref @3k: ~{OFFICIAL_2DGS_SPLAT_REF:,}')
    if splats_now > 0:
        print(f'  vs paper:              {splats_now / OFFICIAL_2DGS_SPLAT_REF:.2f}x')
    near_3k = [r for r in records if 2800 <= r.iteration <= 3200]
    if near_3k:
        r = near_3k[-1]
        print(f'  densify @iter {r.iteration}: {r.n_after:,}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Faster2DGS step profiler + splat count log')
    p.add_argument('-c', '--config', default='../configs/gs_stump_2DGS.yaml', help='Training config yaml')
    p.add_argument('--dataset-path', type=str, default=None, help='Override DATASET.PATH (COLMAP root with sparse/0)')
    p.add_argument('--iters', type=int, default=3500, help='Training iters for splat-count log (0 = skip)')
    p.add_argument('--profile-at', type=int, default=3100, help='Iteration to profile (-1 = skip)')
    p.add_argument('--skip-profile', action='store_true', help='Only run splat-count log')
    p.add_argument('--profile-only', action='store_true', help='Skip splat log')
    p.add_argument('--repeats', type=int, default=30, help='Timed profile repeats')
    p.add_argument('--warmup', type=int, default=5, help='Profile warmup steps')
    p.add_argument('--kernel-only', action='store_true', help='Isolated CUDA kernel timing only (no full step breakdown)')
    p.add_argument('--with-kernel', action='store_true', help='Also run isolated kernel timing before full step breakdown')
    p.add_argument('--csv', type=Path, default=None, help='CSV output (default: output dir / faster2dgs_profile.csv)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print('ERROR: CUDA not available', file=sys.stderr)
        sys.exit(1)

    records: list[DensifyRecord] = []
    output_dir: Path | None = None
    splats_logged = 0

    if not args.profile_only and args.iters > 0:
        trainer, dataset = setup_trainer(args.config, args.dataset_path)
        output_dir = trainer.output_directory
        records = run_training_log(trainer, dataset, args.iters)
        splats_logged = trainer.model.gaussians.means.shape[0]
        print_splat_summary(records, splats_logged)
        Framework.teardown()

    run_profile = args.profile_at >= 0 and not args.skip_profile
    if run_profile:
        trainer, dataset = setup_trainer(args.config, args.dataset_path)
        if output_dir is None:
            output_dir = trainer.output_directory
        if args.kernel_only:
            profile_kernel_only(trainer, dataset, args.profile_at, args.repeats, args.warmup)
        else:
            if args.with_kernel:
                profile_kernel_only(trainer, dataset, args.profile_at, args.repeats, args.warmup)
                trainer, dataset = setup_trainer(args.config, args.dataset_path)
            profile_step(trainer, dataset, args.profile_at, args.repeats, args.warmup)
        if not records:
            print_splat_summary([], trainer.model.gaussians.means.shape[0])
        Framework.teardown()

    if records and output_dir is not None:
        csv_path = args.csv or (Path(output_dir) / 'faster2dgs_profile.csv')
        write_csv(csv_path, records, splats_logged)


if __name__ == '__main__':
    main()
