"""Native Faster2DGS surfel rasterization (requires built ``Faster2DGSCudaBackend._C``)."""

from __future__ import annotations

from Methods.FasterGS.FasterGSCudaBackend import RasterizerSettings as SurfelRasterizerSettings

from .Faster2DGSCudaBackend.torch_bindings.surfel_rasterization import (
    diff_rasterize_surfel_with_aux,
    rasterize_surfel_with_aux,
)

__all__ = [
    'SurfelRasterizerSettings',
    'diff_rasterize_surfel_with_aux',
    'rasterize_surfel_with_aux',
    'has_surfel_backend',
    'configure_backend',
    # Deprecated aliases kept for older scripts.
    'has_native_surfel_backend',
    'has_true_surfel_backend',
    'has_native_diff_surfel_backend',
]


def has_surfel_backend() -> bool:
    """True when the in-repo Faster2DGSCudaBackend surfel extension is importable."""
    return True


def has_native_surfel_backend() -> bool:
    return has_surfel_backend()


def has_true_surfel_backend() -> bool:
    return has_surfel_backend()


def has_native_diff_surfel_backend() -> bool:
    return has_surfel_backend()


def configure_backend(**_kwargs) -> None:
    """No-op retained for API compatibility; native surfel is the only backend."""
