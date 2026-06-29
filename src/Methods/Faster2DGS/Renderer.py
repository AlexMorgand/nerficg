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
    configure_backend,
    diff_rasterize_surfel_with_aux,
    has_external_surfel_backend,
    has_native_diff_surfel_backend,
    has_native_surfel_backend,
    has_true_surfel_backend,
)
from Methods.Faster2DGS.DiffSurfelBackend.camera_utils import viewspace_normal_to_world
from Methods.Faster2DGS.depth_utils import depth_to_normal
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


def _parse_fastgs_aux(auxiliary_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    """Legacy FasterGS 6-channel aux (alpha, mean depth, depth var, disk normal)."""
    return {
        'rend_alpha': auxiliary_maps[0:1],
        'surf_depth': auxiliary_maps[1:2],
        'rend_dist': auxiliary_maps[2:3],
        'rend_normal': auxiliary_maps[3:6],
    }


def _parse_diff_surfel_allmap(
    allmap: torch.Tensor,
    view: View,
    depth_ratio: float,
    world_view_transform: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Official 2DGS 7-channel allmap → training outputs."""
    depth_expected_num = allmap[0:1]
    rend_alpha = allmap[1:2]
    rend_normal_view = allmap[2:5]
    median_depth = allmap[5:6]
    rend_dist = allmap[6:7]

    alpha_safe = rend_alpha.clamp_min(1e-8)
    depth_expected = (depth_expected_num / alpha_safe).nan_to_num(0.0)
    surf_depth = depth_expected * (1.0 - depth_ratio) + depth_ratio * median_depth

    if world_view_transform is not None:
        rend_normal = viewspace_normal_to_world(rend_normal_view, world_view_transform)
    else:
        from Methods.Faster2DGS.DiffSurfelBackend.camera_utils import build_diff_surfel_camera
        wvt, _, _, _, _ = build_diff_surfel_camera(view)
        rend_normal = viewspace_normal_to_world(rend_normal_view, wvt)

    rend_normal = torch.nn.functional.normalize(rend_normal, dim=0, eps=1e-6).nan_to_num(0.0)
    surf_normal_hwc = depth_to_normal(view, surf_depth)
    surf_normal = surf_normal_hwc.permute(2, 0, 1)
    surf_normal = torch.nn.functional.normalize(surf_normal, dim=0, eps=1e-6).nan_to_num(0.0)

    return {
        'rend_alpha': rend_alpha,
        'surf_depth': surf_depth,
        'rend_dist': rend_dist,
        'rend_normal': rend_normal,
        'surf_normal': surf_normal,
    }


@Framework.Configurable.configure(
    REQUIRE_TRUE_2DGS_KERNEL=False,
    USE_SURFEL_BACKEND_API=True,
    USE_DIFF_SURFEL_BACKEND=False,
    # 2DGS README: depth_ratio=0 (mean) for unbounded/large scenes; 1 (median) for bounded/DTU.
    DEPTH_RATIO=0.0,
    Z_LOG_SCALE_COMPAT=-6.0,
    # GUI debug: extra ``splats`` output (rasterize with uniform opacity, ignoring learned α).
    INCLUDE_SPLAT_DEBUG=False,
    SPLAT_DEBUG_UNIFORM_OPACITY=1.0,
)
class Faster2DGSRenderer(FasterGSRenderer):
    """FasterGS renderer wrapper exposing 2DGS-style training outputs."""

    _aux_stats_logged: bool = False
    REQUIRE_TRUE_2DGS_KERNEL: bool = False

    def __init__(self, model: 'BaseModel') -> None:
        BaseRenderer.__init__(self, model, [Faster2DGSModel])
        configure_backend(use_diff_surfel=self.USE_DIFF_SURFEL_BACKEND)
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('Faster2DGS renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'Faster2DGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        if self.REQUIRE_TRUE_2DGS_KERNEL and not has_true_surfel_backend():
            raise Framework.RendererError(
                'REQUIRE_TRUE_2DGS_KERNEL=True, but no surfel backend was found. '
                'Build Faster2DGSCudaBackend: python scripts/install.py -m Faster2DGS'
            )
        if has_native_surfel_backend():
            Logger.log_info('Faster2DGS using native Faster2DGSCudaBackend surfel rasterizer.')
        elif has_external_surfel_backend():
            Logger.log_info(
                'Faster2DGS using external diff-surfel-rasterization (USE_DIFF_SURFEL_BACKEND debug/parity only).'
            )
        else:
            Logger.log_warning(
                'Faster2DGS falling back to FasterGS 3D bridge — not paper parity. '
                'Build Faster2DGSCudaBackend: python scripts/install.py -m Faster2DGS'
            )

    def _rasterize_training(
        self,
        view: View,
        update_densification_info: bool,
        bg_color: torch.Tensor,
        scale_modifier: float = 1.0,
        uniform_opacity: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING)
        if self.USE_SURFEL_BACKEND_API or self.USE_DIFF_SURFEL_BACKEND:
            rgb, auxiliary_maps, radii = diff_rasterize_surfel_with_aux(
                means=self.model.gaussians.means,
                raw_scales_2d=self.model.gaussians.raw_scales_2d,
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
                sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
                densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
                rasterizer_settings=SurfelRasterizerSettings(*settings),
                view=view,
                z_log_scale_compat=self.Z_LOG_SCALE_COMPAT,
                scale_modifier=scale_modifier,
                uniform_opacity=uniform_opacity,
            )
            self._last_training_radii = radii
            return rgb, auxiliary_maps
        opacities = self.model.gaussians.raw_opacities
        if uniform_opacity is not None:
            opacity_logit = math.log(float(uniform_opacity) / (1.0 - float(uniform_opacity)))
            opacities = torch.full_like(opacities, opacity_logit)
        rgb, auxiliary_maps = diff_rasterize_with_aux(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales_3d_compat,
            rotations=self.model.gaussians.raw_rotations,
            opacities=opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=settings,
        )
        self._last_training_radii = torch.empty(0, device=rgb.device)
        return rgb, auxiliary_maps

    def _apply_training_normal_weights(self, parsed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Match 2DGS: only ``surf_normal`` is alpha-weighted; ``rend_normal`` stays raw."""
        rend_alpha = parsed['rend_alpha']
        parsed['surf_normal'] = parsed['surf_normal'] * rend_alpha.detach()
        return parsed

    def _training_outputs_from_aux(self, view: View, auxiliary_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        if has_native_diff_surfel_backend() and auxiliary_maps.shape[0] == 7:
            parsed = _parse_diff_surfel_allmap(auxiliary_maps, view, self.DEPTH_RATIO)
        else:
            parsed = _parse_fastgs_aux(auxiliary_maps)
            parsed['surf_normal'] = depth_to_world_normal(view, parsed['surf_depth'])
            parsed['surf_normal'] = torch.nn.functional.normalize(parsed['surf_normal'], dim=0, eps=1e-6).nan_to_num(0.0)
            parsed['rend_normal'] = torch.nn.functional.normalize(parsed['rend_normal'], dim=0, eps=1e-6).nan_to_num(0.0)
            parsed['rend_normal'] = parsed['rend_normal'] * parsed['rend_alpha'].detach()
            parsed['surf_normal'] = parsed['surf_normal'] * parsed['rend_alpha'].detach()
            return parsed
        return self._apply_training_normal_weights(parsed)

    def _render_splat_debug(self, view: View) -> torch.Tensor:
        """Rasterize surfels with uniform opacity (ignores learned α) for GUI/debug inspection."""
        bg = torch.zeros_like(view.camera.background_color)
        splats, _ = self._rasterize_training(
            view,
            update_densification_info=False,
            bg_color=bg,
            scale_modifier=self.SCALE_MODIFIER,
            uniform_opacity=float(self.SPLAT_DEBUG_UNIFORM_OPACITY),
        )
        return splats.clamp(0.0, 1.0)

    def _pack_inference_outputs(
        self,
        out_chw: dict[str, torch.Tensor],
        *,
        to_chw: bool,
    ) -> dict[str, torch.Tensor]:
        if to_chw:
            return out_chw
        return {k: v.permute(1, 2, 0) for k, v in out_chw.items()}

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb, auxiliary_maps = self._rasterize_training(view, update_densification_info, bg_color)
        parsed = self._training_outputs_from_aux(view, auxiliary_maps)
        out = {
            'rgb': rgb,
            **parsed,
        }
        if update_densification_info and hasattr(self, '_last_training_radii'):
            out['radii'] = self._last_training_radii
        return out

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        settings = extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING)
        if self.USE_SURFEL_BACKEND_API or self.USE_DIFF_SURFEL_BACKEND:
            rgb, auxiliary_maps = self._rasterize_training(
                view,
                update_densification_info=False,
                bg_color=view.camera.background_color,
                scale_modifier=self.SCALE_MODIFIER,
            )
        else:
            rgb, auxiliary_maps = diff_rasterize_with_aux(
                means=self.model.gaussians.means,
                scales=self.model.gaussians.raw_scales_3d_compat + math.log(max(self.SCALE_MODIFIER, 1e-6)),
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
            backend = 'diff-surfel' if has_native_diff_surfel_backend() else 'bridge'
            if mx < 1e-6:
                Logger.log_warning(f'Faster2DGS auxiliary_maps are all zero ({backend}).')
            else:
                Logger.log_info(
                    f'Faster2DGS auxiliary_maps active ({backend}, {aux.shape[0]} ch, abs max={mx:.4g}).'
                )
            Faster2DGSRenderer._aux_stats_logged = True

        parsed = self._training_outputs_from_aux(view, auxiliary_maps)
        rend_alpha = parsed['rend_alpha'].clamp(0.0, 1.0)
        depth = parsed['surf_depth'].clamp_min(0.0)
        rend_normal = parsed['rend_normal']
        normal_norm = torch.linalg.norm(rend_normal, dim=0, keepdim=True)
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
        if self.INCLUDE_SPLAT_DEBUG:
            out['splats'] = self._render_splat_debug(view)
        return self._pack_inference_outputs(out, to_chw=to_chw)

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], view: View, dataset, index: int) -> dict[str, torch.Tensor]:
        out = super().postprocess_outputs(outputs, view, dataset, index)
        if 'normal' in outputs:
            out['normal'] = outputs['normal'].clamp(0.0, 1.0)
        if 'splats' in outputs:
            out['splats'] = outputs['splats'].clamp(0.0, 1.0)
        return out
