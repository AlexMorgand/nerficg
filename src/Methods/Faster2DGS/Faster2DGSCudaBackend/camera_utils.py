"""Build 2DGS-compatible camera matrices from nerficg views."""

from __future__ import annotations

import math

import torch

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float, device: torch.device) -> torch.Tensor:
    """2DGS ``getProjectionMatrix`` (column-major after transpose in caller)."""
    tan_half_fov_y = math.tan(fov_y * 0.5)
    tan_half_fov_x = math.tan(fov_x * 0.5)
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right
    z_sign = 1.0
    p = torch.zeros((4, 4), dtype=torch.float32, device=device)
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def build_surfel_camera(view: View) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Returns ``(world_view_transform, full_proj_transform, campos, tanfovx, tanfovy)``."""
    if not isinstance(view.camera, PerspectiveCamera):
        raise TypeError('surfel rasterization requires a perspective camera')
    device = view.w2c.device
    w2c = view.w2c.to(device=device, dtype=torch.float32)
    world_view_transform = w2c.T.contiguous()
    fov_x = focal_to_fov(view.camera.focal_x, view.camera.width)
    fov_y = focal_to_fov(view.camera.focal_y, view.camera.height)
    projection_matrix = get_projection_matrix(
        view.camera.near_plane,
        view.camera.far_plane,
        fov_x,
        fov_y,
        device,
    ).T.contiguous()
    full_proj_transform = (world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)).squeeze(0)
    campos = view.position.to(device=device, dtype=torch.float32)
    tanfovx = math.tan(fov_x * 0.5)
    tanfovy = math.tan(fov_y * 0.5)
    return world_view_transform, full_proj_transform, campos, tanfovx, tanfovy


def viewspace_normal_to_world(normal_view: torch.Tensor, world_view_transform: torch.Tensor) -> torch.Tensor:
    """Match official 2DGS ``gaussian_renderer``: view-space normal → world."""
    rot = world_view_transform[:3, :3].T
    return (normal_view.permute(1, 2, 0) @ rot).permute(2, 0, 1)
