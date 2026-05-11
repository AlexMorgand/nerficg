"""Surfel rasterization API used by Faster2DGS.

Backend priority:
1) Local `Faster2DGSCudaBackend` compiled extension
2) Compatibility bridge over FasterGS CUDA
"""

from __future__ import annotations

import warnings

import torch

try:
    from .Faster2DGSCudaBackend.torch_bindings.surfel_rasterization import (
        RasterizerSettings as SurfelRasterizerSettings,
        diff_rasterize_surfel_with_aux as _compiled_diff_rasterize_surfel_with_aux,
        rasterize_surfel_with_aux as _compiled_rasterize_surfel_with_aux,
    )
    _USING_COMPILED_SURFEL_BACKEND = True
except Exception:
    _USING_COMPILED_SURFEL_BACKEND = False
    from Methods.FasterGS.FasterGSCudaBackend import (
        RasterizerSettings as SurfelRasterizerSettings,
        diff_rasterize_with_aux as _diff_rasterize_with_aux_fastgs,
    )


def has_true_surfel_backend() -> bool:
    """Returns whether a non-FasterGS compatibility backend is available."""
    return _USING_COMPILED_SURFEL_BACKEND


def _build_compat_scales(raw_scales_2d: torch.Tensor, z_log_scale: float) -> torch.Tensor:
    """Builds temporary 3D log-scales for compatibility with FasterGS CUDA."""
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
    z_log_scale_compat: float = -6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable surfel rasterization interface (transition implementation).

    Args:
        raw_scales_2d: `(N, 2)` log-scales in local tangent plane coordinates.
        z_log_scale_compat: temporary third log-scale used only for the
            compatibility bridge to FasterGS CUDA.
    """
    if _USING_COMPILED_SURFEL_BACKEND:
        return _compiled_diff_rasterize_surfel_with_aux(
            means=means,
            scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=densification_info,
            rasterizer_settings=rasterizer_settings,
        )
    warnings.warn(
        'No Faster2DGSCudaBackend extension found; using compatibility bridge over FasterGS backend.',
        stacklevel=2,
    )
    raw_scales_compat = _build_compat_scales(raw_scales_2d, z_log_scale_compat)
    return _diff_rasterize_with_aux_fastgs(
        means=means,
        scales=raw_scales_compat,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        densification_info=densification_info,
        rasterizer_settings=rasterizer_settings,
    )


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
    z_log_scale_compat: float = -6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only version of the transition surfel API."""
    if _USING_COMPILED_SURFEL_BACKEND:
        return _compiled_rasterize_surfel_with_aux(
            means=means,
            scales_2d=raw_scales_2d,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            rasterizer_settings=rasterizer_settings,
        )
    return diff_rasterize_surfel_with_aux(
        means=means,
        raw_scales_2d=raw_scales_2d,
        rotations=rotations,
        opacities=opacities,
        sh_coefficients_0=sh_coefficients_0,
        sh_coefficients_rest=sh_coefficients_rest,
        densification_info=torch.empty(0, device=means.device),
        rasterizer_settings=rasterizer_settings,
        z_log_scale_compat=z_log_scale_compat,
    )
