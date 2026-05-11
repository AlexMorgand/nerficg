#include "faster2dgs_rasterization_api.h"
#include "faster2dgs_core_rasterization_api.h"

namespace faster2dgs::rasterization {

namespace {
constexpr float kDefaultZLogScale = -6.0f;

__global__ void expand_scales_2d_to_3d_kernel(
    const float* __restrict__ scales_2d,
    float* __restrict__ scales_3d,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int in_base = 2 * idx;
    const int out_base = 3 * idx;
    scales_3d[out_base + 0] = scales_2d[in_base + 0];
    scales_3d[out_base + 1] = scales_2d[in_base + 1];
    scales_3d[out_base + 2] = kDefaultZLogScale;
}

torch::Tensor expand_scales_2d_to_3d(const torch::Tensor& scales_2d) {
    const int n = scales_2d.size(0);
    auto scales_3d = torch::empty(
        {n, 3},
        torch::TensorOptions().dtype(scales_2d.dtype()).device(scales_2d.device())
    );
    if (n == 0) return scales_3d;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    expand_scales_2d_to_3d_kernel<<<blocks, threads>>>(
        scales_2d.contiguous().data_ptr<float>(),
        scales_3d.data_ptr<float>(),
        n
    );
    return scales_3d;
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
forward_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales_2d,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    // Transitional step towards native 2DGS kernels:
    // keep FastGS execution path but construct the thin third axis in CUDA
    // instead of tensor cat in Python/C++.
    const auto scales = expand_scales_2d_to_3d(scales_2d);
    return faster2dgs_core::rasterization::forward_wrapper(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        bg_color,
        active_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_wrapper(
    torch::Tensor& densification_info,
    const torch::Tensor& grad_image,
    const torch::Tensor& grad_auxiliary_maps,
    const torch::Tensor& image,
    const torch::Tensor& auxiliary_maps,
    const torch::Tensor& means,
    const torch::Tensor& scales_2d,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& primitive_buffers,
    const torch::Tensor& tile_buffers,
    const torch::Tensor& instance_buffers,
    const torch::Tensor& bucket_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const int n_instances,
    const int n_buckets,
    const int instance_primitive_indices_selector)
{
    const auto scales = expand_scales_2d_to_3d(scales_2d);
    auto grad_image_total = grad_image;
    if (grad_auxiliary_maps.defined() && grad_auxiliary_maps.numel() > 0) {
        grad_image_total = grad_image_total + grad_auxiliary_maps.narrow(0, 3, 3);
        auto grad_scalar = grad_auxiliary_maps.narrow(0, 0, 1)
                         + grad_auxiliary_maps.narrow(0, 1, 1)
                         + grad_auxiliary_maps.narrow(0, 2, 1);
        grad_image_total = grad_image_total + grad_scalar.expand_as(grad_image_total);
    }
    auto grads = faster2dgs_core::rasterization::backward_wrapper(
        densification_info,
        grad_image_total,
        image,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_rest,
        primitive_buffers,
        tile_buffers,
        instance_buffers,
        bucket_buffers,
        w2c,
        cam_position,
        bg_color,
        active_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        n_instances,
        n_buckets,
        instance_primitive_indices_selector
    );
    auto grad_means = std::get<0>(grads);
    auto grad_scales = std::get<1>(grads);
    auto grad_rotations = std::get<2>(grads);
    auto grad_opacities = std::get<3>(grads);
    auto grad_sh_coefficients_0 = std::get<4>(grads);
    auto grad_sh_coefficients_rest = std::get<5>(grads);
    auto grad_scales_2d = grad_scales.narrow(1, 0, 2);
    return {
        grad_means,
        grad_scales_2d,
        grad_rotations,
        grad_opacities,
        grad_sh_coefficients_0,
        grad_sh_coefficients_rest,
    };
}

std::tuple<torch::Tensor, torch::Tensor>
inference_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales_2d,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    auto result = forward_wrapper(
        means,
        scales_2d,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        bg_color,
        active_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing
    );
    return {std::get<0>(result), std::get<1>(result)};
}

}  // namespace faster2dgs::rasterization
