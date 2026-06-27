"""Faster2DGS/Renderer.py"""

import torch
import math

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseRenderer, BaseModel
from Methods.Faster2DGS.Faster2DGSCudaBackend import (
    SurfelRasterizerSettings,
    diff_rasterize_surfel_with_aux,
    has_true_surfel_backend,
)
from Methods.Faster2DGS.Model import Faster2DGSModel
from Methods.FasterGS.Renderer import FasterGSRenderer, extract_settings
from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize_with_aux


def depth_to_world_normal(view: View, depth: torch.Tensor) -> torch.Tensor:
    """Approximates world-space normals from a depth map."""
    if not isinstance(view.camera, PerspectiveCamera):
        return torch.zeros((3, *depth.shape[1:]), device=depth.device, dtype=depth.dtype)
    h, w = depth.shape[1:]
    xs = torch.arange(w, device=depth.device, dtype=depth.dtype)
    ys = torch.arange(h, device=depth.device, dtype=depth.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    x = (grid_x - view.camera.center_x) / view.camera.focal_x
    y = (grid_y - view.camera.center_y) / view.camera.focal_y
    z = depth[0]
    xyz_cam = torch.stack([x * z, y * z, z], dim=-1)  # H, W, 3
    rot = view.rotation.to(device=depth.device, dtype=depth.dtype)
    xyz_world = xyz_cam @ rot.T + view.position.to(device=depth.device, dtype=depth.dtype)
    dpx = torch.roll(xyz_world, shifts=-1, dims=1) - torch.roll(xyz_world, shifts=1, dims=1)
    dpy = torch.roll(xyz_world, shifts=-1, dims=0) - torch.roll(xyz_world, shifts=1, dims=0)
    normal = torch.cross(dpx, dpy, dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
    normal = normal.permute(2, 0, 1)
    return normal.nan_to_num(0.0)


@Framework.Configurable.configure(
    REQUIRE_TRUE_2DGS_KERNEL=False,
    USE_SURFEL_BACKEND_API=True,
    Z_LOG_SCALE_COMPAT=-6.0,
)
class Faster2DGSRenderer(FasterGSRenderer):
    """FasterGS renderer wrapper exposing 2DGS-style training outputs."""

    _aux_stats_logged: bool = False
    REQUIRE_TRUE_2DGS_KERNEL: bool = False

    def __init__(self, model: 'BaseModel') -> None:
        BaseRenderer.__init__(self, model, [Faster2DGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('Faster2DGS renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'Faster2DGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        if self.REQUIRE_TRUE_2DGS_KERNEL and not has_true_surfel_backend():
            raise Framework.RendererError(
                'REQUIRE_TRUE_2DGS_KERNEL=True, but no true surfel backend was found. '
                'Build/install Faster2DGSCudaBackend.'
            )

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> dict[str, torch.Tensor]:
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING)
        if self.USE_SURFEL_BACKEND_API:
            rgb, auxiliary_maps = diff_rasterize_surfel_with_aux(
                means=self.model.gaussians.means,
                raw_scales_2d=self.model.gaussians.raw_scales[:, :2],
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
                rasterizer_settings=SurfelRasterizerSettings(*settings),
                z_log_scale_compat=self.Z_LOG_SCALE_COMPAT,
            )
        else:
            rgb, auxiliary_maps = diff_rasterize_with_aux(
                means=self.model.gaussians.means,
                scales=self.model.gaussians.raw_scales,
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
                rasterizer_settings=settings,
            )
        rend_alpha = auxiliary_maps[0:1]
        surf_depth = auxiliary_maps[1:2]
        rend_dist = auxiliary_maps[2:3]
        rend_normal = auxiliary_maps[3:6]
        rend_normal = torch.nn.functional.normalize(rend_normal, dim=0, eps=1e-6)
        rend_normal = rend_normal.nan_to_num(0.0)
        surf_normal = depth_to_world_normal(view, surf_depth)
        surf_normal = torch.nn.functional.normalize(surf_normal, dim=0, eps=1e-6)
        surf_normal = surf_normal.nan_to_num(0.0)
        rend_normal = rend_normal * rend_alpha.detach()
        surf_normal = surf_normal * rend_alpha.detach()
        return {
            'rgb': rgb,
            'rend_alpha': rend_alpha,
            'surf_depth': surf_depth,
            'rend_dist': rend_dist,
            'rend_normal': rend_normal,
            'surf_normal': surf_normal,
        }

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING)
        if self.USE_SURFEL_BACKEND_API:
            rgb, auxiliary_maps = diff_rasterize_surfel_with_aux(
                means=self.model.gaussians.means,
                raw_scales_2d=self.model.gaussians.raw_scales[:, :2] + math.log(max(self.SCALE_MODIFIER, 1e-6)),
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=torch.empty(0),
                rasterizer_settings=SurfelRasterizerSettings(*settings),
                z_log_scale_compat=self.Z_LOG_SCALE_COMPAT,
            )
        else:
            rgb, auxiliary_maps = diff_rasterize_with_aux(
                means=self.model.gaussians.means,
                scales=self.model.gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=torch.empty(0),
                rasterizer_settings=settings,
            )
        if not Faster2DGSRenderer._aux_stats_logged:
            aux = auxiliary_maps.detach()
            mx = float(aux.abs().max().cpu())
            if mx < 1e-6:
                Logger.log_warning(
                    'Faster2DGS auxiliary_maps are all zero — rebuild FasterGSCudaBackend or check the scene; '
                    'grey normals usually mean dead aux buffers (zero normals encode to 0.5).'
                )
            else:
                # Channels: alpha [0,1], depth (scene units), depth variance, normal xyz in [-1,1].
                Logger.log_info(
                    f'Faster2DGS auxiliary_maps active (abs max={mx:.4g}; depth/normals are not clamped to [0,1]).'
                )
            Faster2DGSRenderer._aux_stats_logged = True

        rend_alpha = auxiliary_maps[0:1].clamp(0.0, 1.0)
        depth = auxiliary_maps[1:2].clamp_min(0.0)
        rend_normal = auxiliary_maps[3:6]
        normal_norm = torch.linalg.norm(rend_normal, dim=0, keepdim=True)
        # Mean norm is misleading when most pixels are background; max detects stale/zero aux maps.
        if normal_norm.max() < 1e-6:
            rend_normal = depth_to_world_normal(view, depth)
        else:
            rend_normal = torch.nn.functional.normalize(rend_normal, dim=0, eps=1e-6)
        rend_normal = rend_normal.nan_to_num(0.0)
        normal_vis = (rend_normal * 0.5 + 0.5).clamp(0.0, 1.0)
        out = {
            'rgb': rgb.clamp(0.0, 1.0),
            'alpha': rend_alpha,
            'depth': depth,
            'normal': normal_vis,
        }
        if to_chw:
            return out
        return {k: v.permute(1, 2, 0) for k, v in out.items()}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], view: View, dataset, index: int) -> dict[str, torch.Tensor]:
        out = super().postprocess_outputs(outputs, view, dataset, index)
        if 'normal' in outputs:
            out['normal'] = outputs['normal'].clamp(0.0, 1.0)
        return out
