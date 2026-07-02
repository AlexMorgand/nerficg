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
    from Faster2DGSCudaBackend import _C as _backend  # noqa: F401
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__) from e

from .camera_utils import (
    build_surfel_camera,
    focal_to_fov,
    get_projection_matrix,
    viewspace_normal_to_world,
)
from .surfel_rasterization import (
    SurfelAuxMode,
    SurfelRasterizerSettings,
    configure_backend,
    diff_rasterize_surfel_with_aux,
    has_native_diff_surfel_backend,
    has_native_surfel_backend,
    has_surfel_backend,
    has_true_surfel_backend,
    rasterize_surfel,
    rasterize_surfel_with_aux,
)

__all__ = [
    'SurfelAuxMode',
    'SurfelRasterizerSettings',
    'build_surfel_camera',
    'configure_backend',
    'diff_rasterize_surfel_with_aux',
    'focal_to_fov',
    'get_projection_matrix',
    'has_native_diff_surfel_backend',
    'has_native_surfel_backend',
    'has_surfel_backend',
    'has_true_surfel_backend',
    'rasterize_surfel',
    'rasterize_surfel_with_aux',
    'viewspace_normal_to_world',
]
