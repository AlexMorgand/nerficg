#!/usr/bin/env python3
"""Render one test view from a trained official 2DGS checkpoint (run with official env).

Example:
    python official_2dgs_render_test_view.py \\
        --source /path/to/kitchen --model /path/to/kitchen_7k \\
        --iteration 7000 --view-idx 0 --out /tmp/official_view0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', type=Path, default=Path.home() / 'Simulon' / 'dev' / '2d-gaussian-splatting')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', default='images_2')
    parser.add_argument('--iteration', type=int, default=7000)
    parser.add_argument('--view-idx', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    root = args.official_root.resolve()
    if not (root / 'scene').is_dir():
        print(f'missing official 2DGS at {root}', file=sys.stderr)
        return 1
    sys.path.insert(0, str(root))

    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser, Namespace
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    # Build minimal namespace matching train.py layout
    dummy = ArgumentParser()
    mp = ModelParams(dummy)
    pp = PipelineParams(dummy)
    ns = Namespace(
        sh_degree=3,
        source_path=str(args.source.resolve()),
        model_path=str(args.model.resolve()),
        images=args.images,
        resolution=-1,
        white_background=False,
        data_device='cuda',
        eval=True,
        render_items=['RGB'],
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )
    dataset = mp.extract(ns)
    pipe = pp.extract(ns)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    test_cams = scene.getTestCameras()
    if args.view_idx >= len(test_cams):
        print(f'view-idx {args.view_idx} >= {len(test_cams)} test cameras', file=sys.stderr)
        return 1
    cam = test_cams[args.view_idx]
    bg = torch.zeros(3, device='cuda')
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg)
        rgb = pkg['render'].clamp(0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(rgb, str(args.out))
    print(f'wrote {args.out} ({rgb.shape[-1]}x{rgb.shape[-2]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
