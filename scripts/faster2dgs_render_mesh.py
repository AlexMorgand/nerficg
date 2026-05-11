#! /usr/bin/env python3

"""faster2dgs_render_mesh.py: Export Faster2DGS buffers and fuse a TSDF mesh."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Datasets.utils import save_image


def _require_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError as e:
        raise Framework.InferenceError(
            'Open3D is required for meshing. Install it in your environment (e.g. pip install open3d).'
        ) from e


@torch.no_grad()
def main(
    *,
    base_dir: Path,
    checkpoint_name: str,
    subset: str,
    voxel_size: float,
    depth_trunc: float,
    alpha_threshold: float,
    max_frames: int,
    export_buffers: bool,
) -> None:
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    if Framework.config.GLOBAL.METHOD_TYPE != 'Faster2DGS':
        raise Framework.InferenceError(f'Expected METHOD_TYPE=Faster2DGS, got {Framework.config.GLOBAL.METHOD_TYPE}')

    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    dataset.set_mode(subset)
    if len(dataset) == 0:
        raise Framework.InferenceError(f'No views available in subset "{subset}"')

    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
    ).eval()
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    o3d = _require_open3d()
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=5.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    output_root = base_dir / 'faster2dgs_export'
    output_root.mkdir(parents=True, exist_ok=True)
    if export_buffers:
        for name in ('rgb', 'alpha', 'depth', 'normal'):
            (output_root / name).mkdir(exist_ok=True)

    n_frames = len(dataset) if max_frames <= 0 else min(len(dataset), max_frames)
    Logger.log_info(f'processing {n_frames} frames from subset "{subset}"')

    logged_stats = False
    for idx, view in enumerate(Logger.log_progress(dataset, total=n_frames, desc='fuse', leave=False)):
        if idx >= n_frames:
            break
        outputs = renderer.render_image(view, to_chw=True)
        rgb = outputs['rgb'].clamp(0.0, 1.0)
        depth = outputs.get('depth', None)
        alpha = outputs.get('alpha', None)
        normal = outputs.get('normal', None)
        if depth is None:
            raise Framework.InferenceError('Renderer did not provide depth output, cannot fuse TSDF.')

        if not logged_stats:
            a = outputs.get('alpha', None)
            n = outputs.get('normal', None)
            Logger.log_info(
                f'buffer stats (frame 0): depth max={float(depth.max().item()):.4f} mean={float(depth.mean().item()):.6f} '
                + (f'alpha max={float(a.max().item()):.4f} ' if a is not None else 'alpha=missing ')
                + (f'normal mean={float(n.mean().item()):.4f}' if n is not None else 'normal=missing')
            )
            logged_stats = True

        depth = depth.clone()
        if alpha is not None:
            depth[alpha < alpha_threshold] = 0.0
        depth = depth.clamp(0.0, depth_trunc)

        if export_buffers:
            save_image(output_root / 'rgb' / f'{idx:05d}.png', rgb)
            if alpha is not None:
                save_image(output_root / 'alpha' / f'{idx:05d}.png', alpha.expand_as(rgb))
            save_image(output_root / 'depth' / f'{idx:05d}.png', (depth / max(depth_trunc, 1e-6)).expand_as(rgb))
            if normal is not None:
                save_image(output_root / 'normal' / f'{idx:05d}.png', normal.clamp(0.0, 1.0))

        rgb_np = np.ascontiguousarray((rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
        depth_np = np.ascontiguousarray(depth[0].cpu().numpy().astype(np.float32))
        rgb_o3d = o3d.geometry.Image(rgb_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d,
            depth=depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(view.camera.width),
            int(view.camera.height),
            float(view.camera.focal_x),
            float(view.camera.focal_y),
            float(view.camera.center_x),
            float(view.camera.center_y),
        )
        extrinsic = view.w2c.detach().cpu().numpy().astype(np.float64)
        tsdf.integrate(rgbd, intrinsic, extrinsic)

    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = output_root / f'mesh_{subset}_{model.num_iterations_trained}.ply'
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    Logger.log_info(f'mesh written to {mesh_path}')

    Framework.teardown()


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='faster2dgs_render_mesh.py',
        description='Export Faster2DGS buffers and run bounded TSDF meshing.'
    )
    parser.add_argument('-d', '--dir', dest='base_dir', required=True, help='Path to training output directory.')
    parser.add_argument('--checkpoint', dest='checkpoint_name', default='final.pt', help='Checkpoint filename in output/checkpoints.')
    parser.add_argument('--subset', default='train', help='Dataset subset to fuse (train/test/...).')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='TSDF voxel size in scene units.')
    parser.add_argument('--depth_trunc', type=float, default=8.0, help='Maximum depth used for TSDF.')
    parser.add_argument('--alpha_threshold', type=float, default=0.05, help='Minimum alpha for valid depth fusion.')
    parser.add_argument('--max_frames', type=int, default=-1, help='Max number of frames to process (-1 means all).')
    parser.add_argument('--no_export_buffers', action='store_true', help='Disable saving rgb/depth/normal/alpha images.')
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        base_dir=Path(args.base_dir),
        checkpoint_name=args.checkpoint_name,
        subset=args.subset,
        voxel_size=args.voxel_size,
        depth_trunc=args.depth_trunc,
        alpha_threshold=args.alpha_threshold,
        max_frames=args.max_frames,
        export_buffers=not args.no_export_buffers,
    )
