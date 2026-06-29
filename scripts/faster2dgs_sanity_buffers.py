#! /usr/bin/env python3
"""Quick Faster2DGS buffer diagnostic: scales, normals, depth for meshing vs 2DGS."""

from argparse import ArgumentParser
from pathlib import Path

import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Methods.Faster2DGS.Gaussians2D import Gaussians2D
    from Methods.Faster2DGS.Renderer import depth_to_world_normal
    from Logging import Logger


def main(config_path: str, checkpoint: str | None, train_iters: int) -> None:
    Framework.setup(config_path=config_path, require_custom_config=True)
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )
    dataset.set_mode('train')

    if checkpoint:
        model = MI.get_model(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            checkpoint=checkpoint,
        ).eval()
        Logger.log_info(f'loaded checkpoint: {checkpoint}')
    else:
        Framework.config.TRAINING.NUM_ITERATIONS = train_iters
        Framework.config.TRAINING.GUI.ACTIVATE = False
        Framework.config.TRAINING.WANDB.ACTIVATE = False
        Framework.config.TRAINING.BACKUP.FINAL_CHECKPOINT = False
        Framework.config.TRAINING.BACKUP.RENDER_TESTSET = False
        Framework.config.TRAINING.BACKUP.INTERMEDIATE_RENDERINGS = False
        Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 1
        trainer = MI.get_training_instance(method=Framework.config.GLOBAL.METHOD_TYPE, checkpoint=None)
        Logger.log_info(f'running {train_iters} training iterations (fresh Gaussians2D init)')
        trainer.run(dataset)
        model = trainer.model.eval()
        Logger.log_info(f'finished training: n={model.gaussians.means.shape[0]} scales={tuple(model.gaussians.raw_scales.shape)}')

    _log_buffer_sanity(dataset, model)


@torch.no_grad()
def _log_buffer_sanity(dataset, model) -> None:
    g = model.gaussians
    Logger.log_info(f'primitive type: {type(g).__name__}')
    Logger.log_info(f'scale storage shape: {tuple(g.raw_scales.shape)} (expect N×2 for Gaussians2D)')
    if isinstance(g, Gaussians2D):
        Logger.log_info(f'scale XY mean={float(g.scales.mean()):.6f} max={float(g.scales.max()):.6f}')

    renderer = MI.get_renderer(method=Framework.config.GLOBAL.METHOD_TYPE, model=model)
    view = dataset[0]
    train_pkg = renderer.render_image_training(
        view=view, update_densification_info=False, bg_color=view.camera.background_color
    )
    infer_pkg = renderer.render_image(view, to_chw=True)

    rend_n = train_pkg['rend_normal']
    surf_n = train_pkg['surf_normal']
    depth = train_pkg['surf_depth']
    alpha = train_pkg['rend_alpha']
    rend_dist = train_pkg['rend_dist']

    # Masked foreground stats
    fg = (alpha.squeeze(0) > 0.1)
    if fg.any():
        # rend_normal and surf_normal are already alpha-weighted in Renderer.
        rn = torch.nn.functional.normalize(rend_n, dim=0, eps=1e-6)[:, fg]
        sn = torch.nn.functional.normalize(surf_n, dim=0, eps=1e-6)[:, fg]
        cos_disk_vs_surf = (rn * sn).sum(0).clamp(-1, 1)
        Logger.log_info(
            f'foreground cos(rend_normal, surf_normal): mean={float(cos_disk_vs_surf.mean()):.4f} '
            f'median={float(cos_disk_vs_surf.median()):.4f} '
            f'(<0.9 fraction: {float((cos_disk_vs_surf < 0.9).float().mean()):.2%})'
        )
        depth_fg = depth.squeeze(0)[fg]
        Logger.log_info(
            f'depth (mesh input): min={float(depth_fg.min()):.4f} max={float(depth_fg.max()):.4f} '
            f'mean={float(depth_fg.mean()):.4f}'
        )
        Logger.log_info(
            f'rend_dist (depth variance proxy, not 2DGS distortion): mean={float(rend_dist[0, fg].mean()):.6f}'
        )

    Logger.log_info(
        '2DGS: native surfel allmap uses 7 channels (expected depth, alpha, normal, median, distortion). '
        'Bridge fallback: rend_normal = disk axis R[:,2]; rend_dist = depth variance.'
    )
    Framework.teardown()


if __name__ == '__main__':
    parser = ArgumentParser(prog='faster2dgs_sanity_buffers.py')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--checkpoint', default=None, help='Optional .pt checkpoint path')
    parser.add_argument('--train_iters', type=int, default=50, help='Fresh-init training iters if no checkpoint')
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(args.config, args.checkpoint, args.train_iters)
