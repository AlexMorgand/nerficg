#! /usr/bin/env python3

"""faster2dgs_render_mesh.py: Export Faster2DGS buffers and fuse a TSDF mesh (2DGS mesh parity)."""

from argparse import ArgumentParser
from contextlib import contextmanager
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
    from Methods.Faster2DGS.mesh_utils import (
        TsdfParams,
        collect_valid_depth_values,
        filter_depth_by_scene_sphere,
        mask_depth_for_fusion,
        open3d_pinhole_from_view,
        post_process_mesh,
        refine_depth_trunc_from_depths,
        resolve_tsdf_params,
    )


def _require_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError as e:
        raise Framework.InferenceError(
            'Open3D is required for meshing. Install it in your environment (e.g. pip install open3d).'
        ) from e


@contextmanager
def _diffuse_only_sh(model):
    """Match 2DGS mesh export: fuse with DC / diffuse colors only (``active_sh_degree = 0``)."""
    gaussians = model.gaussians
    saved_degree = gaussians.active_sh_degree
    saved_bases = gaussians.active_sh_bases
    gaussians.active_sh_degree = 0
    gaussians.active_sh_bases = 1
    try:
        yield
    finally:
        gaussians.active_sh_degree = saved_degree
        gaussians.active_sh_bases = saved_bases


@torch.no_grad()
def main(
    *,
    base_dir: Path,
    checkpoint_name: str,
    subset: str,
    voxel_size: float,
    depth_trunc: float,
    sdf_trunc: float,
    mesh_res: int,
    alpha_threshold: float,
    use_gt_mask: bool,
    num_cluster: int,
    max_frames: int,
    export_buffers: bool,
    skip_post_process: bool,
    unbounded: bool,
    depth_ratio: float | None,
    config_overrides: list[str],
    sphere_filter: bool,
    sphere_scale: float,
    depth_percentile: float,
    depth_margin: float,
) -> None:
    if unbounded:
        raise Framework.InferenceError(
            'Unbounded mesh extraction (--unbounded) is not implemented yet. '
            'Use bounded TSDF (default) or omit --unbounded.'
        )

    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    for item in config_overrides:
        key, value = item.split('=', 1)
        elements = key.split('.')
        target = Framework.config
        for part in elements[:-1]:
            target = getattr(target, part)
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        setattr(target, elements[-1], value)
    Framework.sync_dataset_external_masks()
    if Framework.config.GLOBAL.METHOD_TYPE != 'Faster2DGS':
        raise Framework.InferenceError(f'Expected METHOD_TYPE=Faster2DGS, got {Framework.config.GLOBAL.METHOD_TYPE}')

    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    dataset.set_mode(subset)
    if len(dataset) == 0:
        raise Framework.InferenceError(f'No views available in subset "{subset}"')

    n_frames = len(dataset) if max_frames <= 0 else min(len(dataset), max_frames)
    fuse_views = [dataset[i] for i in range(n_frames)]
    tsdf_params = resolve_tsdf_params(
        fuse_views,
        depth_trunc=depth_trunc,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        mesh_res=mesh_res,
    )

    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
    ).eval()
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    if depth_ratio is not None:
        renderer.DEPTH_RATIO = float(depth_ratio)
        Logger.log_info(f'mesh export DEPTH_RATIO override: {renderer.DEPTH_RATIO}')
    elif renderer.DEPTH_RATIO == 0.0:
        Logger.log_warning(
            'DEPTH_RATIO=0 (mean depth). Bounded indoor meshing usually needs DEPTH_RATIO=1 '
            '(median depth). Pass --depth_ratio 1 or train with configs/2DGS_mesh.yaml.'
        )

    auto_depth_trunc = depth_trunc <= 0.0
    depth_samples: list[float] = []
    if auto_depth_trunc and depth_percentile > 0.0:
        prescan_views = fuse_views if n_frames <= 48 else fuse_views[:: max(1, n_frames // 48)]
        Logger.log_info(
            f'prescanning {len(prescan_views)}/{n_frames} views for auto depth_trunc '
            f'(percentile={depth_percentile:.2f}, margin={depth_margin})'
        )
        with _diffuse_only_sh(model):
            for view in Logger.log_progress(prescan_views, total=len(prescan_views), desc='depth prescan', leave=False):
                outputs = renderer.render_image(view, to_chw=True)
                depth = outputs.get('depth', None)
                if depth is None:
                    continue
                depth = mask_depth_for_fusion(
                    depth,
                    view,
                    alpha_threshold=alpha_threshold,
                    use_gt_mask=use_gt_mask,
                    rendered_alpha=outputs.get('alpha', None),
                )
                depth_samples.extend(
                    collect_valid_depth_values(
                        depth,
                        outputs.get('alpha', None),
                        alpha_threshold=alpha_threshold,
                        depth_trunc=tsdf_params.depth_trunc,
                    )
                )
        refined_trunc = refine_depth_trunc_from_depths(
            radius=tsdf_params.radius,
            camera_heuristic_trunc=tsdf_params.depth_trunc,
            depth_samples=depth_samples,
            percentile=depth_percentile,
            margin=depth_margin,
        )
        resolved_voxel = voxel_size if voxel_size > 0.0 else refined_trunc / max(mesh_res, 1)
        resolved_sdf = sdf_trunc if sdf_trunc > 0.0 else 5.0 * resolved_voxel
        tsdf_params = TsdfParams(
            center=tsdf_params.center,
            radius=tsdf_params.radius,
            depth_trunc=refined_trunc,
            voxel_size=resolved_voxel,
            sdf_trunc=resolved_sdf,
        )
        Logger.log_info(
            f'refined TSDF params: depth_trunc={tsdf_params.depth_trunc:.4f}, '
            f'voxel_size={tsdf_params.voxel_size:.6f}, sdf_trunc={tsdf_params.sdf_trunc:.6f}'
        )

    sphere_max_distance = sphere_scale * tsdf_params.radius if sphere_filter else 0.0
    if sphere_filter:
        Logger.log_info(
            f'scene-sphere depth filter: max distance={sphere_max_distance:.4f} '
            f'(center radius * {sphere_scale:.2f}, 2DGS uses depth_trunc=2*radius)'
        )

    o3d = _require_open3d()
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_params.voxel_size,
        sdf_trunc=tsdf_params.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    output_root = base_dir / 'faster2dgs_export'
    output_root.mkdir(parents=True, exist_ok=True)
    if export_buffers:
        for name in ('rgb', 'alpha', 'depth', 'normal', 'surf_normal'):
            (output_root / name).mkdir(exist_ok=True)

    Logger.log_info(f'processing {n_frames} frames from subset "{subset}" (diffuse-only SH for fusion)')

    logged_stats = False
    with _diffuse_only_sh(model):
        for idx, view in enumerate(Logger.log_progress(fuse_views, total=n_frames, desc='fuse', leave=False)):
            outputs = renderer.render_image(view, to_chw=True)
            rgb = outputs['rgb'].clamp(0.0, 1.0)
            depth = outputs.get('depth', None)
            alpha = outputs.get('alpha', None)
            normal = outputs.get('normal', None)
            surf_normal = outputs.get('surf_normal', None)
            if depth is None:
                raise Framework.InferenceError('Renderer did not provide depth output, cannot fuse TSDF.')

            if not logged_stats:
                a = outputs.get('alpha', None)
                n = outputs.get('normal', None)
                Logger.log_info(
                    f'buffer stats (frame 0): depth max={float(depth.max().item()):.4f} '
                    f'mean={float(depth.mean().item()):.6f} '
                    + (f'alpha max={float(a.max().item()):.4f} ' if a is not None else 'alpha=missing ')
                    + (f'normal mean={float(n.mean().item()):.4f}' if n is not None else 'normal=missing')
                )
                logged_stats = True

            depth = mask_depth_for_fusion(
                depth,
                view,
                alpha_threshold=alpha_threshold,
                use_gt_mask=use_gt_mask,
                rendered_alpha=alpha,
            )
            depth = filter_depth_by_scene_sphere(
                depth,
                view,
                tsdf_params.center,
                sphere_max_distance,
            )
            depth = depth.clamp(0.0, tsdf_params.depth_trunc)

            if export_buffers:
                save_image(output_root / 'rgb' / f'{idx:05d}.png', rgb)
                if alpha is not None:
                    save_image(output_root / 'alpha' / f'{idx:05d}.png', alpha.expand_as(rgb))
                save_image(
                    output_root / 'depth' / f'{idx:05d}.png',
                    (depth / max(tsdf_params.depth_trunc, 1e-6)).expand_as(rgb),
                )
                if normal is not None:
                    save_image(output_root / 'normal' / f'{idx:05d}.png', normal.clamp(0.0, 1.0))
                if surf_normal is not None:
                    save_image(output_root / 'surf_normal' / f'{idx:05d}.png', surf_normal.clamp(0.0, 1.0))

            rgb_np = np.ascontiguousarray((rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
            depth_np = np.ascontiguousarray(depth[0].cpu().numpy().astype(np.float32))
            rgb_o3d = o3d.geometry.Image(rgb_np)
            depth_o3d = o3d.geometry.Image(depth_np)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d,
                depth=depth_o3d,
                depth_scale=1.0,
                depth_trunc=tsdf_params.depth_trunc,
                convert_rgb_to_intensity=False,
            )
            intrinsic, extrinsic = open3d_pinhole_from_view(view)
            tsdf.integrate(rgbd, intrinsic, extrinsic)

    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    stem = f'mesh_{subset}_{model.num_iterations_trained}'
    mesh_path = output_root / f'{stem}.ply'
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    Logger.log_info(f'mesh written to {mesh_path}')

    if not skip_post_process:
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        mesh_post.compute_vertex_normals()
        mesh_post_path = output_root / f'{stem}_post.ply'
        o3d.io.write_triangle_mesh(str(mesh_post_path), mesh_post)
        Logger.log_info(f'post-processed mesh written to {mesh_post_path}')

    Framework.teardown()


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='faster2dgs_render_mesh.py',
        description='Export Faster2DGS buffers and run bounded TSDF meshing (2DGS parity).'
    )
    parser.add_argument('-d', '--dir', dest='base_dir', required=True, help='Path to training output directory.')
    parser.add_argument('--checkpoint', dest='checkpoint_name', default='final.pt', help='Checkpoint filename in output/checkpoints.')
    parser.add_argument('--subset', default='train', help='Dataset subset to fuse (train/test/...).')
    parser.add_argument(
        '--voxel_size', type=float, default=-1.0,
        help='TSDF voxel size in scene units. <=0 auto: depth_trunc / mesh_res (2DGS default).',
    )
    parser.add_argument(
        '--depth_trunc', type=float, default=-1.0,
        help='Max depth for TSDF. <=0 auto: 2 * estimated camera bounding radius.',
    )
    parser.add_argument(
        '--sdf_trunc', type=float, default=-1.0,
        help='TSDF truncation distance. <=0 auto: 5 * voxel_size.',
    )
    parser.add_argument(
        '--mesh_res', type=int, default=1024,
        help='Grid resolution used to derive auto voxel_size (2DGS default: 1024).',
    )
    parser.add_argument(
        '--alpha_threshold', type=float, default=0.05,
        help='Rendered-alpha cutoff when no GT mask is available.',
    )
    parser.add_argument(
        '--no_gt_mask', action='store_true',
        help='Do not zero depth using dataset alpha/segmentation masks.',
    )
    parser.add_argument(
        '--num_cluster', type=int, default=50,
        help='Keep triangles in the largest N connected components (post-process).',
    )
    parser.add_argument(
        '--skip_post_process', action='store_true',
        help='Skip floater removal (2DGS fuse.ply only, no fuse_post.ply).',
    )
    parser.add_argument(
        '--unbounded', action='store_true',
        help='Unbounded TSDF (MipNeRF360-style). Not implemented yet.',
    )
    parser.add_argument('--max_frames', type=int, default=-1, help='Max frames to process (-1 = all).')
    parser.add_argument(
        '--depth_ratio',
        type=float,
        default=None,
        help='Override renderer DEPTH_RATIO for fusion (bounded indoor: 1.0, unbounded: 0.0).',
    )
    parser.add_argument(
        '--no_sphere_filter',
        action='store_true',
        help='Disable world-space bounding-sphere depth filtering (2DGS uses camera depth_trunc only).',
    )
    parser.add_argument(
        '--sphere_scale',
        type=float,
        default=2.0,
        help='Keep backprojected points within sphere_scale * estimated scene radius (default 2.0, matches 2DGS depth_trunc factor).',
    )
    parser.add_argument(
        '--depth_percentile',
        type=float,
        default=0.99,
        help='When depth_trunc is auto, tighten using this rendered-depth quantile (<=0 disables prescan).',
    )
    parser.add_argument(
        '--depth_margin',
        type=float,
        default=1.05,
        help='Multiplier applied to the depth percentile when refining auto depth_trunc.',
    )
    parser.add_argument('--no_export_buffers', action='store_true', help='Disable saving rgb/depth/normal/alpha PNGs.')
    args, unknown = parser.parse_known_args()
    config_overrides = list(unknown)
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        base_dir=Path(args.base_dir),
        checkpoint_name=args.checkpoint_name,
        subset=args.subset,
        voxel_size=args.voxel_size,
        depth_trunc=args.depth_trunc,
        sdf_trunc=args.sdf_trunc,
        mesh_res=args.mesh_res,
        alpha_threshold=args.alpha_threshold,
        use_gt_mask=not args.no_gt_mask,
        num_cluster=args.num_cluster,
        max_frames=args.max_frames,
        export_buffers=not args.no_export_buffers,
        skip_post_process=args.skip_post_process,
        unbounded=args.unbounded,
        depth_ratio=args.depth_ratio,
        config_overrides=config_overrides,
        sphere_filter=not args.no_sphere_filter,
        sphere_scale=args.sphere_scale,
        depth_percentile=args.depth_percentile,
        depth_margin=args.depth_margin,
    )
