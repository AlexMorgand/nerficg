"""Depth → surface normal for 2DGS geometry losses."""

from __future__ import annotations

import torch

from Datasets.utils import View
from Methods.Faster2DGS.Faster2DGSCudaBackend.camera_utils import build_surfel_camera


def depths_to_points(view: View, depthmap: torch.Tensor) -> torch.Tensor:
    """Unproject depth map to world-space 3D points (2DGS ``depths_to_points``)."""
    world_view_transform, full_proj_transform, _, _, _ = build_surfel_camera(view)
    c2w = world_view_transform.T.inverse()
    w, h = view.camera.width, view.camera.height
    device = depthmap.device
    dtype = depthmap.dtype
    ndc2pix = torch.tensor([
        [w / 2, 0, 0, w / 2],
        [0, h / 2, 0, h / 2],
        [0, 0, 0, 1],
    ], dtype=dtype, device=device).T
    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(w, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        indexing='xy',
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    return depthmap.reshape(-1, 1) * rays_d + rays_o


def depth_to_normal(view: View, depth: torch.Tensor) -> torch.Tensor:
    """World-space normals from depth (2DGS ``depth_to_normal``). Returns ``(H, W, 3)``."""
    if depth.dim() == 3:
        depth_map = depth.squeeze(0)
    else:
        depth_map = depth
    h, w = depth_map.shape
    world_view_transform, _, _, tanfovx, tanfovy = build_surfel_camera(view)
    c2w = world_view_transform.T.inverse()
    rot = c2w[:3, :3]
    origin = c2w[:3, 3]

    fx = w / (2.0 * tanfovx)
    fy = h / (2.0 * tanfovy)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    u = torch.arange(w, device=depth_map.device, dtype=depth_map.dtype)
    v = torch.arange(h, device=depth_map.device, dtype=depth_map.dtype)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    x_cam = (uu - cx) / fx * depth_map
    y_cam = (vv - cy) / fy * depth_map
    z_cam = depth_map
    points = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    points_world = points @ rot.T + origin

    output = torch.zeros_like(points_world)
    dx = points_world[2:, 1:-1] - points_world[:-2, 1:-1]
    dy = points_world[1:-1, 2:] - points_world[1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
