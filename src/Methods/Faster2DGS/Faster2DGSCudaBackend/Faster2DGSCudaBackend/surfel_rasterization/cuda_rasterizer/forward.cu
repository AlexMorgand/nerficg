/*
 * Faster2DGS native surfel rasterizer (NeRFICG).
 * Bucket-checkpoint orchestration from FastGS; surfel forward/backward implements
 * the 2D Gaussian Splatting perspective-disk formulation.
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
#if SURFEL_TILE_CIRCLE_CULL
	tiles_touched[idx] = count_touched_tiles_circle(point_image, static_cast<float>(radius), rect_min, rect_max);
#else
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
#endif
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Surfel ray-splat intersection +
// 7-channel aux, with FastGS-style per-bucket (32 primitive) checkpoints
// of the running accumulators (color, T, depth, normal, M1, M2) so the
// backward can be done bucket-parallel instead of per-tile serial.
template <uint32_t CHANNELS, int AUX_LEVEL>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const uint32_t* __restrict__ tile_bucket_offsets,
	float* __restrict__ final_T,
	float* __restrict__ final_M1,
	float* __restrict__ final_M2,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ median_contrib,
	uint32_t* __restrict__ max_contrib,
	uint32_t* __restrict__ bucket_tile_index,
	float4* __restrict__ bucket_color_T,
	float4* __restrict__ bucket_aux_dn,
	float2* __restrict__ bucket_aux_m,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};
	const uint32_t thread_rank = block.thread_rank();
	const uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	const uint32_t n_pixels = (uint32_t)W * (uint32_t)H;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[tile_id];
	const int n_points_total = range.y - range.x;
	const int rounds = ((n_points_total + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = n_points_total;

	// Bucket bookkeeping: first global bucket of this tile (exclusive prefix).
	const int n_buckets_tile = (n_points_total + SURFEL_BUCKET_SIZE - 1) / SURFEL_BUCKET_SIZE;
	uint32_t bucket_offset = (tile_id == 0) ? 0u : tile_bucket_offsets[tile_id - 1];
	for (int rem = n_buckets_tile, cur = thread_rank; rem > 0; rem -= BLOCK_SIZE, cur += BLOCK_SIZE)
		if (cur < n_buckets_tile) bucket_tile_index[bucket_offset + cur] = tile_id;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	uint32_t median_contributor = 0; // 0 = none; otherwise 1-based contributor index

	// running count of primitives this pixel-thread has stepped through (for bucket checkpoint)
	uint32_t bucket_count = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + thread_rank;
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[thread_rank] = coll_id;
			collected_xy[thread_rank] = points_xy_image[coll_id];
			collected_normal_opacity[thread_rank] = normal_opacity[coll_id];
			collected_Tu[thread_rank] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[thread_rank] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[thread_rank] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Checkpoint running accumulators every SURFEL_BUCKET_SIZE primitives
			// (BEFORE blending this primitive), matching the bucket boundaries.
			if (bucket_count % SURFEL_BUCKET_SIZE == 0)
			{
				const uint32_t slot = bucket_offset * BLOCK_SIZE + thread_rank;
				bucket_color_T[slot] = make_float4(C[0], C[1], C[2], T);
				if constexpr (AUX_LEVEL >= 1) {
					bucket_aux_dn[slot] = make_float4(D, N[0], N[1], N[2]);
					bucket_aux_m[slot] = make_float2(M1, M2);
				}
				bucket_offset++;
			}
			bucket_count++;

			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
			float rho = min(rho3d, rho2d);

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			if (depth < near_n) continue;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
			if constexpr (AUX_LEVEL >= 1) {
				float A = 1-T;
				float m = far_n / (far_n - near_n) * (1 - near_n / depth);
				distortion += (m * m * A + M2 - 2 * m * M1) * w;
				D  += depth * w;
				M1 += m * w;
				M2 += m * m * w;

				if constexpr (AUX_LEVEL >= 2) {
					if (T > 0.5) {
						median_depth = depth;
						median_contributor = contributor; // 1-based
					}
					for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
				}
			}

			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			last_contributor = contributor;
		}
	}

	// Per-tile max contributor (bound for bucket-parallel backward launch)
	__shared__ uint32_t sh_max_contrib;
	if (thread_rank == 0) sh_max_contrib = 0;
	block.sync();
	atomicMax(&sh_max_contrib, last_contributor);
	block.sync();
	if (thread_rank == 0) max_contrib[tile_id] = sh_max_contrib;

	if (inside)
	{
		final_T[pix_id] = T;
		if constexpr (AUX_LEVEL >= 1) {
			final_M1[pix_id] = M1;
			final_M2[pix_id] = M2;
		}
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * n_pixels + pix_id] = C[ch] + T * bg_color[ch];

		if constexpr (AUX_LEVEL >= 1) {
			if constexpr (AUX_LEVEL >= 2) {
				median_contrib[pix_id] = median_contributor;
			}
			out_others[pix_id + DEPTH_OFFSET * n_pixels] = D;
			out_others[pix_id + ALPHA_OFFSET * n_pixels] = 1 - T;
			if constexpr (AUX_LEVEL >= 2) {
				for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * n_pixels] = N[ch];
				out_others[pix_id + MIDDEPTH_OFFSET * n_pixels] = median_depth;
			}
			out_others[pix_id + DISTORTION_OFFSET * n_pixels] = distortion;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	const uint32_t* tile_bucket_offsets,
	float* final_T,
	float* final_M1,
	float* final_M2,
	uint32_t* n_contrib,
	uint32_t* median_contrib,
	uint32_t* max_contrib,
	uint32_t* bucket_tile_index,
	float4* bucket_color_T,
	float4* bucket_aux_dn,
	float2* bucket_aux_m,
	const float* bg_color,
	float* out_color,
	float* out_others,
	int aux_mode)
{
	if (aux_mode == AUX_PHOTOMETRIC) {
		renderCUDA<NUM_CHANNELS, 0> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			means2D,
			colors,
			transMats,
			depths,
			normal_opacity,
			tile_bucket_offsets,
			final_T,
			final_M1,
			final_M2,
			n_contrib,
			median_contrib,
			max_contrib,
			bucket_tile_index,
			bucket_color_T,
			bucket_aux_dn,
			bucket_aux_m,
			bg_color,
			out_color,
			out_others);
	} else if (aux_mode == AUX_DISTORTION) {
		renderCUDA<NUM_CHANNELS, 1> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			means2D,
			colors,
			transMats,
			depths,
			normal_opacity,
			tile_bucket_offsets,
			final_T,
			final_M1,
			final_M2,
			n_contrib,
			median_contrib,
			max_contrib,
			bucket_tile_index,
			bucket_color_T,
			bucket_aux_dn,
			bucket_aux_m,
			bg_color,
			out_color,
			out_others);
	} else {
		renderCUDA<NUM_CHANNELS, 2> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			means2D,
			colors,
			transMats,
			depths,
			normal_opacity,
			tile_bucket_offsets,
			final_T,
			final_M1,
			final_M2,
			n_contrib,
			median_contrib,
			max_contrib,
			bucket_tile_index,
			bucket_color_T,
			bucket_aux_dn,
			bucket_aux_m,
			bg_color,
			out_color,
			out_others);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
