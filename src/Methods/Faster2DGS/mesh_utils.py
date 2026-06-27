"""Faster2DGS/mesh_utils.py: TSDF mesh helpers aligned with official 2DGS ``utils/mesh_utils.py``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

from Datasets.utils import View, get_supervision_alpha
from Logging import Logger

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
