"""Python adapter around official ``diff_surfel_rasterization`` for Faster2DGS."""

from __future__ import annotations

import torch

from Methods.Faster2DGS.DiffSurfelBackend.camera_utils import build_diff_surfel_camera
from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings as NerficgRasterizerSettings

try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, _RasterizeGaussians
except ImportError as e:
    raise ImportError(
        'diff_surfel_rasterization is not installed. Run: '
        'python scripts/install.py -e submodules/diff-surfel-rasterization'
    ) from e


# Official 2DGS allmap layout (7 × H × W).
ALLMAP_DEPTH_EXPECTED = 0
ALLMAP_ALPHA = 1
ALLMAP_NORMAL = slice(2, 5)
ALLMAP_MEDIAN_DEPTH = 5
ALLMAP_DISTORTION = 6
ALLMAP_CHANNELS = 7


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
    width: int,
    height: int,
) -> None:
    """Accumulate view-space mean gradient norms (official 2DGS ``add_densification_stats``).

    Do not apply FasterGS/3DGS pixel scaling here — diff-surfel ``means2D`` grads are already in
    the same units as official ``viewspace_points.grad``, and pixel scaling makes ~100% of
    Gaussians pass ``densify_grad_threshold`` every cycle.
    """
    del width, height  # kept for call-site compatibility
    if densification_info.numel() == 0 or grad_means2d is None:
        return
    visible = radii > 0
    if not bool(visible.any().item()):
        return
    densification_info[0, visible] += 1.0
    densification_info[1, visible] += grad_means2d[visible, :2].norm(dim=1)


def _register_densification_hook(
    screenspace_points: torch.Tensor,
    densification_info: torch.Tensor,
    radii: torch.Tensor,
    width: int,
    height: int,
) -> None:
    """Register hook on screenspace means — fires after full backward (official 2DGS timing)."""

    def _screenspace_grad_hook(grad: torch.Tensor) -> torch.Tensor:
        _update_densification_info(densification_info, radii, grad, width, height)
        return grad

    screenspace_points.register_hook(_screenspace_grad_hook)


def _build_settings(
    rasterizer_settings: NerficgRasterizerSettings,
    view,
    scale_modifier: float,
) -> GaussianRasterizationSettings:
    world_view_transform, full_proj_transform, campos, tanfovx, tanfovy = build_diff_surfel_camera(view)
    sh_degree = max(0, int(round(rasterizer_settings.active_sh_bases ** 0.5)) - 1)
    return GaussianRasterizationSettings(
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
        prefiltered=False,
        debug=False,
    )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native diff-surfel forward; returns ``(rgb, allmap, radii)`` with 7-channel allmap."""
    settings = _build_settings(rasterizer_settings, view, scale_modifier)
    sh = _pack_sh(sh_coefficients_0, sh_coefficients_rest, rasterizer_settings.active_sh_bases)
    scales_2d = raw_scales_2d.exp()
    rots = torch.nn.functional.normalize(rotations, dim=-1)
    opacities_act = opacities.sigmoid()
    if uniform_opacity is not None:
        opacities_act = torch.full_like(opacities_act, float(uniform_opacity))
    empty = torch.empty(0, device=means.device)

    screenspace_points = torch.zeros_like(means, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    rgb, radii, allmap = _RasterizeGaussians.apply(
        means,
        screenspace_points,
        sh,
        empty,
        opacities_act,
        scales_2d,
        rots,
        empty,
        settings,
    )

    if densification_info.numel() > 0:
        _register_densification_hook(
            screenspace_points,
            densification_info,
            radii,
            rasterizer_settings.width,
            rasterizer_settings.height,
        )

    return rgb, allmap, radii


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
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(False):
        rgb, allmap, _ = diff_rasterize_surfel_with_aux(
            means=means,
            raw_scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=torch.empty(0, device=means.device),
            rasterizer_settings=rasterizer_settings,
            view=view,
            scale_modifier=scale_modifier,
        )
    return rgb.detach(), allmap.detach()
