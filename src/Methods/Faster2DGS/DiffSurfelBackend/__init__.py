"""Install hook for the official diff-surfel-rasterization CUDA extension."""

from pathlib import Path
import sys

import Framework

extension_dir = Path(__file__).resolve().parents[4] / 'submodules' / 'diff-surfel-rasterization'
__extension_name__ = 'diff_surfel_rasterization'
__install_command__ = [
    sys.executable, '-m', 'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',
]

try:
    import diff_surfel_rasterization  # noqa: F401
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__) from e

from Methods.Faster2DGS.DiffSurfelBackend.rasterization import (
    ALLMAP_ALPHA,
    ALLMAP_CHANNELS,
    ALLMAP_DEPTH_EXPECTED,
    ALLMAP_DISTORTION,
    ALLMAP_MEDIAN_DEPTH,
    ALLMAP_NORMAL,
    diff_rasterize_surfel_with_aux,
    rasterize_surfel_with_aux,
)

__all__ = [
    'ALLMAP_ALPHA',
    'ALLMAP_CHANNELS',
    'ALLMAP_DEPTH_EXPECTED',
    'ALLMAP_DISTORTION',
    'ALLMAP_MEDIAN_DEPTH',
    'ALLMAP_NORMAL',
    'diff_rasterize_surfel_with_aux',
    'rasterize_surfel_with_aux',
]
