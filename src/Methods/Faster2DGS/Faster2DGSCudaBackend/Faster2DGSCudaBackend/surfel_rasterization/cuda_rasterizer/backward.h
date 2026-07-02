/*
 * Faster2DGS native surfel rasterizer (NeRFICG).
 * Bucket-checkpoint orchestration from FastGS; surfel forward/backward implements
 * the 2D Gaussian Splatting perspective-disk formulation.
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		int n_buckets,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const float* transMats,
		const float* colors,
		const float* depths,
		const float* out_color,
		const float* out_others,
		const float* final_T,
		const float* final_M1,
		const float* final_M2,
		const uint32_t* n_contrib,
		const uint32_t* median_contrib,
		const uint32_t* max_contrib,
		const uint32_t* tile_bucket_offsets,
		const uint32_t* bucket_tile_index,
		const float4* bucket_color_T,
		const float4* bucket_aux_dn,
		const float2* bucket_aux_m,
		const float* dL_dpixels,
		const float* dL_depths,
		float * dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		int aux_mode = 2);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
