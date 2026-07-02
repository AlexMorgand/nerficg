"""Native Faster2DGS surfel rasterization (7-channel allmap, analytic aux backward)."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

from Methods.Faster2DGS.Faster2DGSCudaBackend.camera_utils import build_surfel_camera
from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings as NerficgRasterizerSettings

try:
    from Faster2DGSCudaBackend import _C
except ImportError as e:
    raise ImportError(
        'Faster2DGSCudaBackend surfel extension is not built. Run:\n'
        '  python scripts/install.py -m Faster2DGS'
    ) from e


ALLMAP_DEPTH_EXPECTED = 0
ALLMAP_ALPHA = 1
ALLMAP_NORMAL = slice(2, 5)
ALLMAP_MEDIAN_DEPTH = 5
ALLMAP_DISTORTION = 6
ALLMAP_CHANNELS = 7


class SurfelAuxMode:
    """Must match AUX_* constants in cuda_rasterizer/auxiliary.h."""

    PHOTOMETRIC = 0
    DISTORTION = 1
    FULL = 2


def _resolve_aux_mode(*, aux_mode: int | None, photometric_only: bool) -> int:
    if aux_mode is not None:
        return int(aux_mode)
    return SurfelAuxMode.PHOTOMETRIC if photometric_only else SurfelAuxMode.FULL


class SurfelRasterizerSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool = False
    debug: bool = False
    aux_mode: int = SurfelAuxMode.FULL


def _pack_sh(
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    active_sh_bases: int,
) -> torch.Tensor:
    sh = torch.cat([sh_coefficients_0, sh_coefficients_rest], dim=1)
    if sh.shape[1] > active_sh_bases:
        sh = sh[:, :active_sh_bases]
    return sh.contiguous()


def _update_densification_info(
    densification_info: torch.Tensor,
    radii: torch.Tensor,
    grad_means2d: torch.Tensor,
) -> None:
    if densification_info.numel() == 0 or grad_means2d is None:
        return
    visible = radii > 0
    if not bool(visible.any().item()):
        return
    densification_info[0, visible] += 1.0
    densification_info[1, visible] += grad_means2d[visible].norm(dim=-1)


def _register_densification_hook(
    screenspace_points: torch.Tensor,
    densification_info: torch.Tensor,
    radii: torch.Tensor,
) -> None:
    def _screenspace_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        _update_densification_info(densification_info, radii, grad)
        return grad

    screenspace_points.register_hook(_screenspace_grad_hook)


def _build_settings(
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float,
    aux_mode: int,
) -> SurfelRasterizerSettings:
    world_view_transform, full_proj_transform, campos, tanfovx, tanfovy = build_surfel_camera(view)
    sh_degree = max(0, int(round(rasterizer_settings.active_sh_bases ** 0.5)) - 1)
    return SurfelRasterizerSettings(
        image_height=rasterizer_settings.height,
        image_width=rasterizer_settings.width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=rasterizer_settings.bg_color,
        scale_modifier=scale_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=sh_degree,
        campos=campos,
        aux_mode=aux_mode,
    )


class _RasterizeSurfels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        opacities: torch.Tensor,
        scales_2d: torch.Tensor,
        rotations: torch.Tensor,
        settings: SurfelRasterizerSettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = torch.empty(0, device=means3D.device)
        num_rendered, color, allmap, radii, geom_buffer, binning_buffer, img_buffer = _C.rasterize_gaussians(
            settings.bg,
            means3D,
            empty,
            opacities,
            scales_2d,
            rotations,
            settings.scale_modifier,
            empty,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            sh,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.debug,
            settings.aux_mode,
        )
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            empty, means3D, scales_2d, rotations, empty, radii, sh,
            geom_buffer, binning_buffer, img_buffer, color, allmap,
        )
        ctx.means2D = means2D
        return color, radii, allmap

    @staticmethod
    def backward(
        ctx: Any,
        grad_out_color: torch.Tensor,
        grad_radii: torch.Tensor,
        grad_allmap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        del grad_radii
        settings = ctx.settings
        num_rendered = ctx.num_rendered
        colors_precomp, means3D, scales_2d, rotations, cov3Ds_precomp, radii, sh, geom_buffer, binning_buffer, img_buffer, out_color, out_others = ctx.saved_tensors
        aux_mode = settings.aux_mode
        if aux_mode != SurfelAuxMode.PHOTOMETRIC and (
            grad_allmap.numel() == 0 or float(grad_allmap.abs().max()) == 0.0
        ):
            aux_mode = SurfelAuxMode.PHOTOMETRIC
        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales_2d,
            grad_rotations,
        ) = _C.rasterize_gaussians_backward(
            settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales_2d,
            rotations,
            settings.scale_modifier,
            cov3Ds_precomp,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            grad_out_color,
            grad_allmap,
            sh,
            settings.sh_degree,
            settings.campos,
            geom_buffer,
            num_rendered,
            binning_buffer,
            img_buffer,
            out_color,
            out_others,
            settings.debug,
            aux_mode,
        )
        del grad_colors_precomp, grad_cov3Ds_precomp
        return grad_means3D, grad_means2D, grad_sh, grad_opacities, grad_scales_2d, grad_rotations, None


def diff_rasterize_surfel_with_aux(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float = 1.0,
    uniform_opacity: float | None = None,
    photometric_only: bool = False,
    aux_mode: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if view is None:
        raise ValueError('view is required for native surfel rasterization')
    resolved_aux = _resolve_aux_mode(aux_mode=aux_mode, photometric_only=photometric_only)
    settings = _build_settings(rasterizer_settings, view, scale_modifier, resolved_aux)
    sh = _pack_sh(sh_coefficients_0, sh_coefficients_rest, rasterizer_settings.active_sh_bases)
    scales_2d = raw_scales_2d.exp()
    rots = torch.nn.functional.normalize(rotations, dim=-1)
    opacities_act = opacities.sigmoid()
    if uniform_opacity is not None:
        opacities_act = torch.full_like(opacities_act, float(uniform_opacity))

    screenspace_points = torch.zeros_like(means, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    rgb, radii, allmap = _RasterizeSurfels.apply(
        means,
        screenspace_points,
        sh,
        opacities_act,
        scales_2d,
        rots,
        settings,
    )

    if densification_info.numel() > 0:
        _register_densification_hook(screenspace_points, densification_info, radii)

    return rgb, allmap, radii


def _forward_surfel(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float,
    aux_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct CUDA forward (no autograd). Used for inference / benchmark."""
    settings = _build_settings(rasterizer_settings, view, scale_modifier, aux_mode)
    sh = _pack_sh(sh_coefficients_0, sh_coefficients_rest, rasterizer_settings.active_sh_bases)
    scales_2d = raw_scales_2d.exp()
    rots = torch.nn.functional.normalize(rotations, dim=-1)
    opacities_act = opacities.sigmoid()
    empty = torch.empty(0, device=means.device)
    _num_rendered, color, allmap, radii, _geom, _bin, _img = _C.rasterize_gaussians(
        settings.bg,
        means,
        empty,
        opacities_act,
        scales_2d,
        rots,
        settings.scale_modifier,
        empty,
        settings.viewmatrix,
        settings.projmatrix,
        settings.tanfovx,
        settings.tanfovy,
        settings.image_height,
        settings.image_width,
        sh,
        settings.sh_degree,
        settings.campos,
        settings.prefiltered,
        settings.debug,
        settings.aux_mode,
    )
    return color, allmap, radii


@torch.no_grad()
def rasterize_surfel(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float = 1.0,
    to_chw: bool = True,
    clamp_output: bool = True,
) -> torch.Tensor:
    """FastGS-style RGB-only inference (photometric kernel, no autograd or aux)."""
    rgb, _allmap, _radii = _forward_surfel(
        means=means,
        raw_scales_2d=raw_scales_2d,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        rasterizer_settings=rasterizer_settings,
        view=view,
        scale_modifier=scale_modifier,
        aux_mode=SurfelAuxMode.PHOTOMETRIC,
    )
    if clamp_output:
        rgb = rgb.clamp(0.0, 1.0)
    return rgb if to_chw else rgb.permute(1, 2, 0)


@torch.no_grad()
def rasterize_surfel_with_aux(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float = 1.0,
    aux_mode: int = SurfelAuxMode.FULL,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb, allmap, _radii = _forward_surfel(
        means=means,
        raw_scales_2d=raw_scales_2d,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        rasterizer_settings=rasterizer_settings,
        view=view,
        scale_modifier=scale_modifier,
        aux_mode=aux_mode,
    )
    return rgb, allmap
