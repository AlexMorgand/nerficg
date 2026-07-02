#!/usr/bin/env python3
"""Native Faster2DGS surfel kernel smoke + timing benchmark."""

import time

import utils

with utils.DiscoverSourcePath():
    import torch
    import Framework
    from Implementations import Methods as MI, Datasets as DI
    from Methods.Faster2DGS.Faster2DGSCudaBackend import diff_rasterize_surfel_with_aux, SurfelRasterizerSettings
    from Methods.FasterGS.Renderer import extract_settings


def rel_err(a, b):
    denom = b.abs().mean().item() + 1e-8
    return (a - b).abs().mean().item() / denom


def time_fwd_bwd(g, view, settings, iters=30):
    dens = torch.zeros(2, g.means.shape[0], device='cuda')

    def step():
        for p in (g._means, g._scales, g._rotations, g._opacities, g._sh_coefficients_0, g._sh_coefficients_rest):
            if p.grad is not None:
                p.grad = None
        rgb, allmap, radii = diff_rasterize_surfel_with_aux(
            means=g._means, raw_scales_2d=g._scales, rotations=g._rotations,
            opacities=g._opacities, sh_coefficients_0=g._sh_coefficients_0,
            sh_coefficients_rest=g._sh_coefficients_rest, densification_info=dens,
            rasterizer_settings=SurfelRasterizerSettings(*settings), view=view)
        (rgb.mean() + 0.1 * allmap[6].mean() + 0.01 * allmap[:6].mean()).backward()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    Framework.setup(config_path='../configs/2DGS_m360.yaml', require_custom_config=True)
    Framework.config.TRAINING.GUI.ACTIVATE = False
    Framework.config.TRAINING.DATA.PRELOADING_LEVEL = 0
    ds = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer = MI.get_training_instance(Framework.config.GLOBAL.METHOD_TYPE, None)
    for cb in trainer._gather_callbacks(-1):
        cb(trainer, 0, ds)
    g = trainer.model.gaussians
    ds.train()
    view = trainer.train_sampler.get(dataset=ds)['view']
    settings = extract_settings(view, g.active_sh_bases, view.camera.background_color, False)
    print('splats', g.means.shape[0], 'res', view.camera.width, 'x', view.camera.height)

    dens = torch.zeros(2, g.means.shape[0], device='cuda')
    rgb, allmap, radii = diff_rasterize_surfel_with_aux(
        means=g._means, raw_scales_2d=g._scales, rotations=g._rotations,
        opacities=g._opacities, sh_coefficients_0=g._sh_coefficients_0,
        sh_coefficients_rest=g._sh_coefficients_rest, densification_info=dens,
        rasterizer_settings=SurfelRasterizerSettings(*settings), view=view)
    loss = rgb.mean() + 0.1 * allmap[6].mean()
    loss.backward()
    print('forward rgb range', float(rgb.min()), float(rgb.max()))
    print('grad means', float(g._means.grad.abs().sum()))
    print('NaN rgb', bool(torch.isnan(rgb).any()), 'NaN grad', bool(torch.isnan(g._means.grad).any()))

    ms = time_fwd_bwd(g, view, settings)
    print(f'=== native surfel fwd+bwd === {ms:.1f} ms ({1000.0 / ms:.1f} kernel-steps/s)')


if __name__ == '__main__':
    main()
