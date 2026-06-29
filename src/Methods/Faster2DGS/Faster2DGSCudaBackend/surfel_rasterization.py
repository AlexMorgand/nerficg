"""Surfel rasterization API used by Faster2DGS.

Default (``use_diff_surfel=False``): in-repo ``Faster2DGSCudaBackend`` surfel kernel.

``use_diff_surfel=True`` is a debug/parity switch only — loads the external
``diff_surfel_rasterization`` pip package for A/B comparison, not for production.

Last resort: FasterGS 3D compatibility bridge (6-ch aux, not paper parity).
"""

from __future__ import annotations

import math
import warnings

import torch

from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings as SurfelRasterizerSettings

_USING_NATIVE_SURFEL_BACKEND = False
_USING_EXTERNAL_DIFF_SURFEL = False
_native_diff_rasterize = None
_native_rasterize = None
_external_diff_rasterize = None
_external_rasterize = None
_diff_rasterize_with_aux_fastgs = None


def _try_native_backend() -> bool:
    global _native_diff_rasterize, _native_rasterize, _USING_NATIVE_SURFEL_BACKEND
    try:
        from .Faster2DGSCudaBackend.torch_bindings.surfel_rasterization import (
            diff_rasterize_surfel_with_aux as _compiled_diff,
            rasterize_surfel_with_aux as _compiled_inf,
        )
    except ImportError:
        return False
    _native_diff_rasterize = _compiled_diff
    _native_rasterize = _compiled_inf
    _USING_NATIVE_SURFEL_BACKEND = True
    return True


def _try_external_backend() -> bool:
    global _external_diff_rasterize, _external_rasterize, _USING_EXTERNAL_DIFF_SURFEL
    try:
        from Methods.Faster2DGS.DiffSurfelBackend.rasterization import (
            diff_rasterize_surfel_with_aux as _ds_diff,
            rasterize_surfel_with_aux as _ds_rasterize,
        )
    except ImportError:
        return False
    _external_diff_rasterize = _ds_diff
    _external_rasterize = _ds_rasterize
    _USING_EXTERNAL_DIFF_SURFEL = True
    return True


def configure_backend(*, use_diff_surfel: bool = False) -> None:
    """Select rasterizer backend (call from renderer ``__init__``)."""
    global _USING_NATIVE_SURFEL_BACKEND, _USING_EXTERNAL_DIFF_SURFEL
    global _native_diff_rasterize, _native_rasterize
    global _external_diff_rasterize, _external_rasterize
    global _diff_rasterize_with_aux_fastgs

    _USING_NATIVE_SURFEL_BACKEND = False
    _USING_EXTERNAL_DIFF_SURFEL = False
    _native_diff_rasterize = None
    _native_rasterize = None
    _external_diff_rasterize = None
    _external_rasterize = None
    _diff_rasterize_with_aux_fastgs = None

    if use_diff_surfel:
        if _try_external_backend():
            return
        warnings.warn(
            'USE_DIFF_SURFEL_BACKEND=True but diff_surfel_rasterization is not installed. '
            'Install for A/B only: python scripts/install.py -e submodules/diff-surfel-rasterization',
            stacklevel=2,
        )
    elif _try_native_backend():
        return
    else:
        warnings.warn(
            'Faster2DGSCudaBackend surfel extension is not available; '
            'falling back to FasterGS 3D bridge (not 2DGS parity). Build with:\n'
            '  python scripts/install.py -m Faster2DGS',
            stacklevel=2,
        )

    from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize_with_aux as _fastgs_diff
    _diff_rasterize_with_aux_fastgs = _fastgs_diff


def has_true_surfel_backend() -> bool:
    """True when native or external surfel backend is active."""
    return _USING_NATIVE_SURFEL_BACKEND or _USING_EXTERNAL_DIFF_SURFEL


def has_native_surfel_backend() -> bool:
    """True when Faster2DGSCudaBackend surfel extension is active."""
    return _USING_NATIVE_SURFEL_BACKEND


def has_external_surfel_backend() -> bool:
    """True when the external diff-surfel pip package is active (debug/parity only)."""
    return _USING_EXTERNAL_DIFF_SURFEL


def has_native_diff_surfel_backend() -> bool:
    """Deprecated alias — True for native or external 7-ch surfel backends."""
    return has_true_surfel_backend()


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
    if _USING_NATIVE_SURFEL_BACKEND or _USING_EXTERNAL_DIFF_SURFEL:
        backend = _native_diff_rasterize if _USING_NATIVE_SURFEL_BACKEND else _external_diff_rasterize
        if view is None:
            raise ValueError('view is required for surfel rasterization')
        return backend(
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
    warnings.warn(
        'No surfel backend found; using compatibility bridge over FasterGS backend.',
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
    if _USING_NATIVE_SURFEL_BACKEND or _USING_EXTERNAL_DIFF_SURFEL:
        backend = _native_rasterize if _USING_NATIVE_SURFEL_BACKEND else _external_rasterize
        if view is None:
            raise ValueError('view is required for surfel rasterization')
        return backend(
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
