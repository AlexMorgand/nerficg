"""Surfel rasterization API used by Faster2DGS.

Backend priority:
1) Official ``diff_surfel_rasterization`` (Phase C — native 2DGS)
2) Local ``Faster2DGSCudaBackend`` compiled extension (Phase A bridge)
3) Compatibility bridge over FasterGS CUDA
"""

from __future__ import annotations

import math
import warnings

import torch

from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings as SurfelRasterizerSettings

_USE_DIFF_SURFEL = False
_USING_DIFF_SURFEL_BACKEND = False
_USING_COMPILED_SURFEL_BACKEND = False
_compiled_diff_rasterize = None
_compiled_rasterize = None
_diff_surfel_diff_rasterize = None
_diff_surfel_rasterize = None
_diff_rasterize_with_aux_fastgs = None


def configure_backend(*, use_diff_surfel: bool) -> None:
    """Select rasterizer backend (call from renderer ``__init__``)."""
    global _USE_DIFF_SURFEL, _USING_DIFF_SURFEL_BACKEND, _USING_COMPILED_SURFEL_BACKEND
    global _compiled_diff_rasterize, _compiled_rasterize
    global _diff_surfel_diff_rasterize, _diff_surfel_rasterize, _diff_rasterize_with_aux_fastgs

    _USE_DIFF_SURFEL = bool(use_diff_surfel)
    _USING_DIFF_SURFEL_BACKEND = False
    _USING_COMPILED_SURFEL_BACKEND = False
    _compiled_diff_rasterize = None
    _compiled_rasterize = None
    _diff_surfel_diff_rasterize = None
    _diff_surfel_rasterize = None
    _diff_rasterize_with_aux_fastgs = None

    if _USE_DIFF_SURFEL:
        try:
            from Methods.Faster2DGS.DiffSurfelBackend.rasterization import (
                diff_rasterize_surfel_with_aux as _ds_diff,
                rasterize_surfel_with_aux as _ds_rasterize,
            )
            _diff_surfel_diff_rasterize = _ds_diff
            _diff_surfel_rasterize = _ds_rasterize
            _USING_DIFF_SURFEL_BACKEND = True
            return
        except ImportError:
            warnings.warn(
                'USE_DIFF_SURFEL_BACKEND=True but diff_surfel_rasterization is not installed; '
                'falling back to Faster2DGSCudaBackend. Run: '
                'python scripts/install.py -e submodules/diff-surfel-rasterization',
                stacklevel=2,
            )

    try:
        from .Faster2DGSCudaBackend.torch_bindings.surfel_rasterization import (
            diff_rasterize_surfel_with_aux as _compiled_diff,
            rasterize_surfel_with_aux as _compiled_inf,
        )
        _compiled_diff_rasterize = _compiled_diff
        _compiled_rasterize = _compiled_inf
        _USING_COMPILED_SURFEL_BACKEND = True
    except ImportError:
        from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize_with_aux as _fastgs_diff
        _diff_rasterize_with_aux_fastgs = _fastgs_diff


def has_true_surfel_backend() -> bool:
    """True when native diff-surfel or a compiled surfel extension is active."""
    return _USING_DIFF_SURFEL_BACKEND or _USING_COMPILED_SURFEL_BACKEND


def has_native_diff_surfel_backend() -> bool:
    return _USING_DIFF_SURFEL_BACKEND


def _build_compat_scales(raw_scales_2d: torch.Tensor, z_log_scale: float) -> torch.Tensor:
    if raw_scales_2d.ndim != 2 or raw_scales_2d.shape[1] != 2:
        raise ValueError(f'expected raw_scales_2d of shape (N, 2), got {tuple(raw_scales_2d.shape)}')
    z = torch.full(
        (raw_scales_2d.shape[0], 1),
        fill_value=float(z_log_scale),
        dtype=raw_scales_2d.dtype,
        device=raw_scales_2d.device,
    )
    return torch.cat((raw_scales_2d, z), dim=1)


def diff_rasterize_surfel_with_aux(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: SurfelRasterizerSettings,
    view=None,
    z_log_scale_compat: float = -6.0,
    scale_modifier: float = 1.0,
    uniform_opacity: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_radii = torch.empty(0, device=means.device, dtype=torch.int32)
    if _USING_DIFF_SURFEL_BACKEND:
        if view is None:
            raise ValueError('view is required for diff-surfel rasterization')
        return _diff_surfel_diff_rasterize(
            means=means,
            raw_scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=densification_info,
            rasterizer_settings=rasterizer_settings,
            view=view,
            scale_modifier=scale_modifier,
            uniform_opacity=uniform_opacity,
        )
    if _USING_COMPILED_SURFEL_BACKEND:
        if uniform_opacity is not None:
            opacities = torch.full_like(opacities, math.log(float(uniform_opacity) / (1.0 - float(uniform_opacity))))
        rgb, aux = _compiled_diff_rasterize(
            means=means,
            scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=densification_info,
            rasterizer_settings=rasterizer_settings,
        )
        return rgb, aux, empty_radii
    warnings.warn(
        'No Faster2DGSCudaBackend extension found; using compatibility bridge over FasterGS backend.',
        stacklevel=2,
    )
    raw_scales_compat = _build_compat_scales(raw_scales_2d, z_log_scale_compat)
    if uniform_opacity is not None:
        opacities = torch.full_like(opacities, math.log(float(uniform_opacity) / (1.0 - float(uniform_opacity))))
    rgb, aux = _diff_rasterize_with_aux_fastgs(
        means=means,
        scales=raw_scales_compat,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        densification_info=densification_info,
        rasterizer_settings=rasterizer_settings,
    )
    return rgb, aux, empty_radii


@torch.no_grad()
def rasterize_surfel_with_aux(
    *,
    means: torch.Tensor,
    raw_scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: SurfelRasterizerSettings,
    view=None,
    z_log_scale_compat: float = -6.0,
    scale_modifier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _USING_DIFF_SURFEL_BACKEND:
        if view is None:
            raise ValueError('view is required for diff-surfel rasterization')
        return _diff_surfel_rasterize(
            means=means,
            raw_scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            rasterizer_settings=rasterizer_settings,
            view=view,
            scale_modifier=scale_modifier,
        )
    if _USING_COMPILED_SURFEL_BACKEND:
        return _compiled_rasterize(
            means=means,
            scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            rasterizer_settings=rasterizer_settings,
        )
    rgb, aux, _ = diff_rasterize_surfel_with_aux(
        means=means,
        raw_scales_2d=raw_scales_2d,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        densification_info=torch.empty(0, device=means.device),
        rasterizer_settings=rasterizer_settings,
        view=view,
        z_log_scale_compat=z_log_scale_compat,
        scale_modifier=scale_modifier,
    )
    return rgb, aux
