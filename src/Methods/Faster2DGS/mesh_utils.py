"""Faster2DGS/mesh_utils.py: TSDF mesh helpers aligned with official 2DGS ``utils/mesh_utils.py``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

from Datasets.utils import View, get_supervision_alpha
from Logging import Logger
from Methods.Faster2DGS.depth_utils import depths_to_points

if TYPE_CHECKING:
    import open3d as o3d


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Nearest point to all camera focal axes (2DGS ``render_utils.focus_point_fn``)."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, (0, 2, 1))
    mt_m = np.transpose(m, (0, 2, 1)) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def estimate_bounding_sphere(views: Iterable[View]) -> tuple[np.ndarray, float]:
    """Estimate scene center and radius from camera poses (2DGS ``GaussianExtractor.estimate_bounding_sphere``)."""
    c2ws = np.stack([view.c2w_numpy for view in views], axis=0)
    poses = c2ws[:, :3, :]
    center = focus_point_fn(poses)
    radius = float(np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min())
    radius = max(radius, 1e-4)
    return center, radius


@dataclass(frozen=True)
class TsdfParams:
    center: np.ndarray
    radius: float
    depth_trunc: float
    voxel_size: float
    sdf_trunc: float


def open3d_pinhole_from_view(view: View) -> tuple['o3d.geometry.PinholeCameraIntrinsic', np.ndarray]:
    """Open3D intrinsics/extrinsic matching official 2DGS ``to_cam_open3d``."""
    import open3d as o3d

    from Methods.Faster2DGS.Faster2DGSCudaBackend.camera_utils import (
        build_surfel_camera,
        focal_to_fov,
        get_projection_matrix,
    )

    world_view_transform, _, _, _, _ = build_surfel_camera(view)
    width, height = int(view.camera.width), int(view.camera.height)
    device = world_view_transform.device
    ndc2pix = torch.tensor(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    ).T
    fov_x = focal_to_fov(view.camera.focal_x, width)
    fov_y = focal_to_fov(view.camera.focal_y, height)
    projection_matrix = get_projection_matrix(
        view.camera.near_plane,
        view.camera.far_plane,
        fov_x,
        fov_y,
        device,
    ).T
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrins[0, 0].item()),
        float(intrins[1, 1].item()),
        float(intrins[0, 2].item()),
        float(intrins[1, 2].item()),
    )
    extrinsic = world_view_transform.T.detach().cpu().numpy().astype(np.float64)
    return intrinsic, extrinsic


def resolve_tsdf_params(
    views: Iterable[View],
    *,
    depth_trunc: float,
    voxel_size: float,
    sdf_trunc: float,
    mesh_res: int,
) -> TsdfParams:
    """Resolve TSDF parameters; negative values select 2DGS-style auto defaults."""
    view_list = list(views)
    if not view_list:
        raise ValueError('resolve_tsdf_params requires at least one view')
    center, radius = estimate_bounding_sphere(view_list)
    resolved_depth_trunc = (2.0 * radius) if depth_trunc <= 0.0 else depth_trunc
    resolved_voxel_size = (resolved_depth_trunc / max(mesh_res, 1)) if voxel_size <= 0.0 else voxel_size
    resolved_sdf_trunc = (5.0 * resolved_voxel_size) if sdf_trunc <= 0.0 else sdf_trunc
    Logger.log_info(
        f'estimated bounding radius={radius:.4f}, center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]'
    )
    Logger.log_info(
        f'TSDF params: depth_trunc={resolved_depth_trunc:.4f}, '
        f'voxel_size={resolved_voxel_size:.6f}, sdf_trunc={resolved_sdf_trunc:.6f}'
    )
    Logger.log_info(f'Use depth_trunc >= {2.0 * radius:.4f} for bounded fusion (2DGS heuristic).')
    return TsdfParams(
        center=center,
        radius=radius,
        depth_trunc=resolved_depth_trunc,
        voxel_size=resolved_voxel_size,
        sdf_trunc=resolved_sdf_trunc,
    )


def mask_depth_for_fusion(
    depth: torch.Tensor,
    view: View,
    *,
    alpha_threshold: float,
    use_gt_mask: bool,
    rendered_alpha: torch.Tensor | None,
) -> torch.Tensor:
    """Zero invalid depth pixels before TSDF fusion (GT mask preferred, like 2DGS)."""
    depth = depth.clone()
    if use_gt_mask and (supervision_alpha := get_supervision_alpha(view)) is not None:
        gt_alpha = supervision_alpha
        if gt_alpha.ndim == 3:
            gt_alpha = gt_alpha[0]
        depth[0, gt_alpha < 0.5] = 0.0
        return depth
    if rendered_alpha is not None:
        alpha = rendered_alpha[0] if rendered_alpha.ndim == 3 else rendered_alpha
        depth[0, alpha < alpha_threshold] = 0.0
    return depth


def filter_depth_by_scene_sphere(
    depth: torch.Tensor,
    view: View,
    center: np.ndarray,
    max_distance: float,
) -> torch.Tensor:
    """Zero depth pixels whose world-space backprojection lies outside a bounding sphere.

    Complements camera-space ``depth_trunc`` (2DGS bounded fusion): edge rays and depth
    floaters can sit below ``depth_trunc`` in view-space Z yet lie far from the scene
    center in Euclidean distance, inflating the TSDF volume.
    """
    if max_distance <= 0.0:
        return depth
    depth = depth.clone()
    depth_map = depth[0] if depth.ndim == 3 else depth
    if not bool((depth_map > 0).any().item()):
        return depth

    center_t = torch.tensor(center, device=depth_map.device, dtype=depth_map.dtype)
    points = depths_to_points(view, depth_map)
    dist = torch.linalg.norm(points - center_t, dim=-1)
    valid = dist <= max_distance
    depth_map = depth_map.reshape(-1)
    depth_map[~valid] = 0.0
    if depth.ndim == 3:
        depth[0] = depth_map.reshape(depth.shape[1:])
    else:
        depth = depth_map.reshape(depth.shape)
    return depth


def collect_valid_depth_values(
    depth: torch.Tensor,
    rendered_alpha: torch.Tensor | None,
    *,
    alpha_threshold: float,
    depth_trunc: float,
) -> list[float]:
    """Gather positive depths from a masked frame for auto ``depth_trunc`` estimation."""
    depth_map = depth[0] if depth.ndim == 3 else depth
    mask = depth_map > 0
    if rendered_alpha is not None:
        alpha = rendered_alpha[0] if rendered_alpha.ndim == 3 else rendered_alpha
        mask = mask & (alpha >= alpha_threshold)
    if depth_trunc > 0.0:
        mask = mask & (depth_map <= depth_trunc)
    values = depth_map[mask]
    if values.numel() == 0:
        return []
    return values.detach().cpu().numpy().astype(np.float64).tolist()


def refine_depth_trunc_from_depths(
    *,
    radius: float,
    camera_heuristic_trunc: float,
    depth_samples: list[float],
    percentile: float,
    margin: float,
) -> float:
    """Tighten auto ``depth_trunc`` using rendered depth statistics (bounded scenes).

    Keeps the 2DGS camera heuristic as a floor when data is sparse; otherwise uses
    ``min(heuristic, quantile * margin)`` so distant outliers do not expand the volume.
    """
    if percentile <= 0.0 or not depth_samples:
        return camera_heuristic_trunc
    q = float(np.quantile(np.asarray(depth_samples, dtype=np.float64), percentile))
    data_cap = q * margin
    refined = min(camera_heuristic_trunc, data_cap)
    floor = max(2.0 * radius * 0.85, 1e-4)
    refined = max(refined, floor)
    Logger.log_info(
        f'depth_trunc refine: camera heuristic={camera_heuristic_trunc:.4f}, '
        f'p{int(percentile * 100)}={q:.4f}, margin={margin:.3f} -> {refined:.4f} '
        f'(floor={floor:.4f})'
    )
    return refined


def post_process_mesh(mesh: 'o3d.geometry.TriangleMesh', cluster_to_keep: int = 50) -> 'o3d.geometry.TriangleMesh':
    """Remove floaters by keeping the largest connected triangle clusters (2DGS ``post_process_mesh``)."""
    import open3d as o3d

    if cluster_to_keep <= 0:
        return mesh
    mesh_out = copy.deepcopy(mesh)
    triangle_clusters, cluster_n_triangles, _cluster_area = mesh_out.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) == 0:
        return mesh_out
    n_cluster = np.sort(cluster_n_triangles.copy())[-min(cluster_to_keep, len(cluster_n_triangles))]
    n_cluster = max(int(n_cluster), 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_out.remove_triangles_by_mask(triangles_to_remove)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    Logger.log_info(
        f'mesh post-process: vertices {len(mesh.vertices)} -> {len(mesh_out.vertices)}, '
        f'triangles {len(mesh.triangles)} -> {len(mesh_out.triangles)}'
    )
    return mesh_out
