"""Legacy re-exports — camera math lives in Faster2DGSCudaBackend."""

from Methods.Faster2DGS.Faster2DGSCudaBackend.camera_utils import (
    build_surfel_camera as build_diff_surfel_camera,
    focal_to_fov,
    get_projection_matrix,
    viewspace_normal_to_world,
)

__all__ = [
    'build_diff_surfel_camera',
    'focal_to_fov',
    'get_projection_matrix',
    'viewspace_normal_to_world',
]
