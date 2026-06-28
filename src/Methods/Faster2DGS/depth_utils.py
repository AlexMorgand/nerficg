"""2DGS-compatible depth → surface normal (from hbb1/2d-gaussian-splatting ``utils/point_utils.py``)."""

from __future__ import annotations

import torch

from Datasets.utils import View
from Methods.Faster2DGS.DiffSurfelBackend.camera_utils import build_diff_surfel_camera


def depths_to_points(view: View, depthmap: torch.Tensor) -> torch.Tensor:
    """Unproject depth map to world-space 3D points (2DGS ``depths_to_points``)."""
    world_view_transform, full_proj_transform, _, _, _ = build_diff_surfel_camera(view)
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
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
