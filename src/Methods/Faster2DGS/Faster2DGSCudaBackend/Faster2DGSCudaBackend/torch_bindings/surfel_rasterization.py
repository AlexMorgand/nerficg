from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from Faster2DGSCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor
    cam_position: torch.Tensor
    bg_color: torch.Tensor
    active_sh_bases: int
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    near_plane: float
    far_plane: float
    proper_antialiasing: bool

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
        )


class _RasterizeSurfelWithAux(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales_2d: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            image,
            auxiliary_maps,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales_2d,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            auxiliary_maps,
            means,
            scales_2d,
            rotations,
            opacities,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image, auxiliary_maps

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
        grad_auxiliary_maps: torch.Tensor,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]':
        (
            grad_means, grad_scales_2d, grad_rotations, grad_opacities,
            grad_sh_coefficients_0, grad_sh_coefficients_rest
        ) = _C.backward(
            ctx.densification_info,
            grad_image,
            grad_auxiliary_maps,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales_2d,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            None,
            None,
        )


def diff_rasterize_surfel_with_aux(
    means: torch.Tensor,
    scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RasterizeSurfelWithAux.apply(
        means,
        scales_2d,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        rasterizer_settings,
    )


@torch.no_grad()
def rasterize_surfel_with_aux(
    means: torch.Tensor,
    scales_2d: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.inference(
        means,
        scales_2d,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        *rasterizer_settings.as_tuple(),
    )
