#! /usr/bin/env python3
"""Smoke-test 2DGS densification grad accumulation (run from scripts/)."""

import utils

with utils.DiscoverSourcePath():
    import torch
    import Framework
    from Implementations import Methods as MI, Datasets as DI
    from Methods.Faster2DGS.Faster2DGSCudaBackend import configure_backend


def main() -> None:
    Framework.setup(config_path='../configs/2DGS_bicycle.yaml', require_custom_config=True)
    Framework.config.TRAINING.NUM_ITERATIONS = 650
    Framework.config.TRAINING.GUI.ACTIVATE = False
    Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 1
    configure_backend(use_diff_surfel=Framework.config.RENDERER.USE_DIFF_SURFEL_BACKEND)

    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer = MI.get_training_instance(Framework.config.GLOBAL.METHOD_TYPE, None)

    for cb in trainer._gather_callbacks(-1):
        cb(trainer, 0, dataset)

    gaussians = trainer.model.gaussians
    n_init = gaussians.means.shape[0]
    print(f'initial gaussians: {n_init:,}')

    info_max = 0.0
    for iteration in range(650):
        for cb in trainer._gather_callbacks(0):
            if (
                (cb.start_iteration is not None and iteration < cb.start_iteration)
                or (cb.end_iteration is not None and iteration > cb.end_iteration)
                or (cb.iteration_stride is not None and (iteration - (cb.start_iteration or 0)) % cb.iteration_stride != 0)
            ):
                continue
            cb(trainer, iteration, dataset)

        if gaussians.densification_info is not None:
            info_max = max(info_max, float(gaussians.densification_info[1].max().item()))

        if iteration in (499, 500, 600, 649):
            stats = getattr(gaussians, '_last_densify_stats', None)
            print(
                f'iter {iteration}: n={gaussians.means.shape[0]:,}, '
                f'densify_info_max={info_max:.6f}, stats={stats}'
            )

    print(f'final: {gaussians.means.shape[0]:,} (delta {gaussians.means.shape[0] - n_init:+,})')
    Framework.teardown()


if __name__ == '__main__':
    main()
