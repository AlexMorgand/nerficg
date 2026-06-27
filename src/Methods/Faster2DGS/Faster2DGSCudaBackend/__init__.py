from pathlib import Path
import sys

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    sys.executable, '-m', 'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',
]

try:
    # CUDAExtension is installed as top-level Faster2DGSCudaBackend._C (see setup.py), same as
    # FasterGSCudaBackend — not as Methods....Faster2DGSCudaBackend._C, so never use "from . import _C".
    from Faster2DGSCudaBackend import _C as _backend  # noqa: F401
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__) from e

from .surfel_rasterization import (
    SurfelRasterizerSettings,
    diff_rasterize_surfel_with_aux,
    rasterize_surfel_with_aux,
    has_true_surfel_backend,
)

__all__ = [
    'SurfelRasterizerSettings',
    'diff_rasterize_surfel_with_aux',
    'rasterize_surfel_with_aux',
    'has_true_surfel_backend',
]
