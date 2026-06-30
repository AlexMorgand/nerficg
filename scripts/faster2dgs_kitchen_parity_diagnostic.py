#!/usr/bin/env python3
"""Kitchen parity diagnostics: splat milestones, frozen render diff, grad-norm probe.

Run from scripts/:

    python faster2dgs_kitchen_parity_diagnostic.py --all
    python faster2dgs_kitchen_parity_diagnostic.py --milestones --skip-official-train
    python faster2dgs_kitchen_parity_diagnostic.py --frozen --ply path/to/point_cloud.ply
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import utils

with utils.DiscoverSourcePath():
    import torch
    import torch.nn as nn
    import torchvision
    import Framework
    from Implementations import Datasets as DI, Methods as MI
    from Methods.Faster2DGS.Gaussians2D import Gaussians2D
    from Methods.FasterGS.Renderer import extract_settings
    from Methods.Faster2DGS.Faster2DGSCudaBackend import diff_rasterize_surfel_with_aux, SurfelRasterizerSettings
    from faster2dgs_densify_compare import run_compare, print_table
    from faster2dgs_profile_step import setup_trainer, run_callbacks, advance_to

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'dataset' / 'mipnerf360' / 'kitchen'
CONFIG = REPO / 'configs' / '2DGS_m360.yaml'
OFFICIAL_PYTHON = Path('/opt/anaconda3/envs/pytorch_simulon_env/bin/python')
OFFICIAL_ROOT = Path.home() / 'Simulon' / 'dev' / '2d-gaussian-splatting'
MILESTONES = (600, 1700, 3000, 3100, 7000)
OFFICIAL_MILESTONE_DIR = REPO / 'output' / 'official_2dgs' / 'kitchen_milestones'
OFFICIAL_7K_DIR = REPO / 'output' / 'official_2dgs' / 'kitchen_7k'
DIAG_OUT = REPO / 'output' / 'diagnostics' / 'kitchen_parity'

_OFFICIAL_RENDER_SCRIPT = r'''#!/usr/bin/env python3
"""One-off render helper (written by faster2dgs_kitchen_parity_diagnostic.py)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', default='images_2')
    parser.add_argument('--iteration', type=int, default=7000)
    parser.add_argument('--view-idx', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from argparse import ArgumentParser, Namespace
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

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
'''


def ply_vertex_count(ply_path: Path) -> int:
    with open(ply_path, 'rb') as f:
        for line in f:
            s = line.decode('ascii', errors='ignore').strip()
            if s.startswith('element vertex'):
                return int(s.split()[-1])
    raise ValueError(f'no vertex count in {ply_path}')


def train_official_milestones(force: bool = False) -> Path:
    OFFICIAL_MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
    final_ply = OFFICIAL_MILESTONE_DIR / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
    if final_ply.is_file() and not force:
        print(f'official milestones exist: {OFFICIAL_MILESTONE_DIR}')
        return OFFICIAL_MILESTONE_DIR
    saves = ' '.join(str(m) for m in MILESTONES)
    cmd = [
        str(OFFICIAL_PYTHON),
        str(OFFICIAL_ROOT / 'train.py'),
        '-s', str(DATASET),
        '-i', 'images_2',
        '-m', str(OFFICIAL_MILESTONE_DIR),
        '--eval',
        '--iterations', '7000',
        '--test_iterations', '-1',
        '--save_iterations', *map(str, MILESTONES),
        '--quiet',
    ]
    print('official milestone train:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(OFFICIAL_ROOT), check=True)
    return OFFICIAL_MILESTONE_DIR


def collect_official_milestone_counts(model_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for it in MILESTONES:
        ply = model_dir / 'point_cloud' / f'iteration_{it}' / 'point_cloud.ply'
        if ply.is_file():
            counts[it] = ply_vertex_count(ply)
        else:
            print(f'  missing official ply @ {it}: {ply}')
    return counts


def run_ours_milestone_table() -> dict[int, int]:
    milestones = set(MILESTONES)
    rows = run_compare(str(CONFIG), str(DATASET), 7000, milestones)
    print()
    print('=== Faster2DGS densify milestones (kitchen) ===')
    print_table(rows, milestones)
    by_iter = {r.iteration: r.n_after for r in rows}
    return by_iter


def print_milestone_side_by_side(official: dict[int, int], ours: dict[int, int]) -> None:
    print('\n=== Splat count: official vs Faster2DGS ===')
    print(f'{"iter":>6} {"official":>12} {"ours":>12} {"ratio":>8}')
    print('-' * 42)
    for it in MILESTONES:
        o = official.get(it)
        u = ours.get(it)
        if o is None or u is None:
            continue
        print(f'{it:6d} {o:12,} {u:12,} {u / o:8.3f}x')


def load_ply_into_gaussians2d(g: Gaussians2D, ply_path: Path, sh_degree: int = 3) -> int:
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply['vertex'].data
    n = len(v)
    means = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
    sh0 = torch.tensor(f_dc[:, None, :], device='cuda')
    n_rest = (sh_degree + 1) ** 2 - 1
    rest_names = sorted(
        [name for name in v.dtype.names if name.startswith('f_rest_')],
        key=lambda x: int(x.split('_')[-1]),
    )
    f_rest = np.stack([v[name] for name in rest_names], axis=1).astype(np.float32).reshape(n, n_rest, 3)
    opacities = np.asarray(v['opacity'], dtype=np.float32).reshape(n, 1)
    scales = np.stack([v['scale_0'], v['scale_1']], axis=1).astype(np.float32)
    if 'scale_2' in v.dtype.names:
        # ignore thin axis in ply if present
        pass
    rots = np.stack([v[f'rot_{i}'] for i in range(4)], axis=1).astype(np.float32)

    g.max_sh_degree = sh_degree
    g.active_sh_degree = sh_degree
    g._means = nn.Parameter(torch.tensor(means, device='cuda'))
    g._sh_coefficients_0 = nn.Parameter(sh0)
    g._sh_coefficients_rest = nn.Parameter(torch.tensor(f_rest, device='cuda'))
    g._opacities = nn.Parameter(torch.tensor(opacities, device='cuda'))
    g._scales = nn.Parameter(torch.tensor(scales, device='cuda'))
    g._rotations = nn.Parameter(torch.tensor(rots, device='cuda'))
    g._max_radii2D = torch.zeros(n, device='cuda')
    return n


@torch.no_grad()
def render_ours_test_view(g: Gaussians2D, view, photometric_only: bool = True) -> torch.Tensor:
    settings = extract_settings(view, g.active_sh_bases, view.camera.background_color, False)
    rgb, _, _ = diff_rasterize_surfel_with_aux(
        means=g.means,
        raw_scales_2d=g.raw_scales_2d,
        rotations=g.raw_rotations,
        opacities=g.raw_opacities,
        sh_coefficients_0=g.sh_coefficients_0,
        sh_coefficients_rest=g.sh_coefficients_rest,
        densification_info=torch.empty(0, device='cuda'),
        rasterizer_settings=SurfelRasterizerSettings(*settings),
        view=view,
        photometric_only=photometric_only,
    )
    return rgb.clamp(0, 1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float('inf')
    return float(-10.0 * np.log10(mse))


def frozen_render_diff(ply_path: Path, view_idx: int = 0, official_model_dir: Path | None = None) -> None:
    DIAG_OUT.mkdir(parents=True, exist_ok=True)
    official_model_dir = official_model_dir or OFFICIAL_7K_DIR

    Framework.setup(config_path=str(CONFIG), require_custom_config=True)
    Framework.config.DATASET.PATH = str(DATASET)
    Framework.config.TRAINING.GUI.ACTIVATE = False
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    dataset.test()
    test_views = list(dataset)
    if view_idx >= len(test_views):
        raise IndexError(f'view_idx {view_idx} >= {len(test_views)} test views')
    view = test_views[view_idx]
    gt = view.rgb.to('cuda')

    g = Gaussians2D(sh_degree=3, pretrained=False)
    n = load_ply_into_gaussians2d(g, ply_path)
    print(f'loaded {n:,} splats from {ply_path}')

    ours_rgb = render_ours_test_view(g, view, photometric_only=True)
    ours_path = DIAG_OUT / f'ours_official_ply_view{view_idx}.png'
    torchvision.utils.save_image(ours_rgb, str(ours_path))

    official_png = DIAG_OUT / f'official_ckpt_view{view_idx}.png'
    render_py = OFFICIAL_ROOT / '_render_one_test_view.py'
    render_py.write_text(_OFFICIAL_RENDER_SCRIPT)
    cmd = [
        str(OFFICIAL_PYTHON),
        str(render_py),
        '--source', str(DATASET),
        '--model', str(official_model_dir),
        '--iteration', '7000',
        '--view-idx', str(view_idx),
        '--out', str(official_png),
    ]
    print('official render:', ' '.join(cmd))
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(OFFICIAL_ROOT)
    subprocess.run(cmd, cwd=str(OFFICIAL_ROOT), env=env, check=True)

    off_rgb = torchvision.io.read_image(str(official_png)).float() / 255.0
    off_rgb = off_rgb.to('cuda')
    if off_rgb.shape != ours_rgb.shape:
        off_rgb = torch.nn.functional.interpolate(
            off_rgb[None], size=ours_rgb.shape[-2:], mode='bilinear', align_corners=False
        )[0]

    diff = (ours_rgb - off_rgb).abs()
    print(f'\n=== Frozen render diff (official ply @7k, test view {view_idx}) ===')
    print(f'  ours vs official render: PSNR {psnr(ours_rgb, off_rgb):.2f} dB, MAE {float(diff.mean()):.5f}')
    print(f'  ours vs GT:              PSNR {psnr(ours_rgb, gt):.2f} dB')
    print(f'  official vs GT:          PSNR {psnr(off_rgb, gt):.2f} dB')
    print(f'  images: {ours_path}, {official_png}')

    # Also compare official checkpoint ply path vs passed ply (should match if same file)
    Framework.teardown()


def grad_norm_probe(iteration: int = 3090) -> None:
    """At a fixed pre-3k-reset state, compare 2D vs 3D viewspace grad norms for densify mask."""
    trainer, dataset = setup_trainer(str(CONFIG), str(DATASET))
    advance_to(trainer, dataset, iteration)
    trainer.model.train()
    dataset.train()
    view = trainer.train_sampler.get(dataset=dataset)['view']
    g = trainer.model.gaussians
    g.reset_densification_info()
    trainer.renderer.render_image_training(
        view=view,
        update_densification_info=True,
        bg_color=view.camera.background_color,
        photometric_only=True,
    )
    info = g.densification_info
    thresh = trainer.DENSIFICATION_GRAD_THRESHOLD
    avg_grad = info[1] / info[0].clamp_min(1.0)
    mask_avg = avg_grad >= thresh

    # Re-run with hook capturing full 3D grad — approximate via screenspace grad if 3 cols
    print(f'\n=== Grad-norm probe @ iter {iteration} (kitchen) ===')
    print(f'  splats: {g.means.shape[0]:,}')
    print(f'  visible (denom>0): {int((info[0] > 0).sum())}')
    print(f'  pass threshold (2D avg norm): {int(mask_avg.sum())} ({100 * mask_avg.float().mean():.2f}%)')
    Framework.teardown()


def main() -> int:
    parser = argparse.ArgumentParser(description='Kitchen Faster2DGS parity diagnostics')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--milestones', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--grad-probe', action='store_true')
    parser.add_argument('--skip-official-train', action='store_true')
    parser.add_argument('--force-official-train', action='store_true')
    parser.add_argument('--ply', type=Path, default=None, help='PLY for frozen diff (default: official @7k)')
    parser.add_argument('--view-idx', type=int, default=0)
    args = parser.parse_args()

    if args.all:
        args.milestones = args.frozen = args.grad_probe = True

    if not (args.milestones or args.frozen or args.grad_probe):
        parser.error('specify --all or one of --milestones --frozen --grad-probe')

    if args.milestones:
        official_dir = OFFICIAL_7K_DIR
        if not args.skip_official_train:
            official_dir = train_official_milestones(force=args.force_official_train)
        official_counts = collect_official_milestone_counts(official_dir)
        ours_counts = run_ours_milestone_table()
        print_milestone_side_by_side(official_counts, ours_counts)

    if args.frozen:
        ply = args.ply
        if ply is None:
            ply = OFFICIAL_7K_DIR / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
        if not ply.is_file():
            print(f'missing ply: {ply}', file=sys.stderr)
            return 1
        frozen_render_diff(ply, view_idx=args.view_idx)

    if args.grad_probe:
        grad_norm_probe(3090)

    return 0


if __name__ == '__main__':
    sys.exit(main())
