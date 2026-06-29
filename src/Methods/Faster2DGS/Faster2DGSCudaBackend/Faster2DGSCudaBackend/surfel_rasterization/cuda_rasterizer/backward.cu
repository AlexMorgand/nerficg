/*
 * Faster2DGS native surfel rasterizer (NeRFICG).
 * Bucket-checkpoint orchestration from FastGS; surfel forward/backward implements
 * the 2D Gaussian Splatting perspective-disk formulation.
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure (FastGS bucket-parallel).
//
// One block per bucket, 32 lanes = the 32 (depth-sorted) primitives of that
// bucket. Each lane accumulates gradients for its primitive over all 256 pixels
// of the tile, streamed diagonally via warp shuffles. The per-pixel aux suffix
// sums (color/depth/normal/alpha/distortion contributed by *later* primitives)
// are reconstructed front-to-back from the forward bucket checkpoints + the final
// output maps, avoiding the original 2DGS per-tile serial reverse traversal.
template <uint32_t C>
__global__ void __launch_bounds__(32)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ out_color,
	const float* __restrict__ out_others,
	const float* __restrict__ final_T,
	const float* __restrict__ final_M1,
	const float* __restrict__ final_M2,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ median_contrib,
	const uint32_t* __restrict__ max_contrib,
	const uint32_t* __restrict__ tile_bucket_offsets,
	const uint32_t* __restrict__ bucket_tile_index,
	const float4* __restrict__ bucket_color_T,
	const float4* __restrict__ bucket_aux_dn,
	const float2* __restrict__ bucket_aux_m,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float* __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);
	const uint32_t bucket_idx = block.group_index().x;
	const uint32_t lane = warp.thread_rank();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t tile_id = bucket_tile_index[bucket_idx];
	const uint2 range = ranges[tile_id];
	const int tile_n_primitives = range.y - range.x;
	const uint32_t tile_first_bucket = (tile_id == 0) ? 0u : tile_bucket_offsets[tile_id - 1];
	const int tile_bucket_idx = bucket_idx - tile_first_bucket;
	if (tile_bucket_idx * SURFEL_BUCKET_SIZE >= (int)max_contrib[tile_id]) return;

	const int tile_primitive_idx = tile_bucket_idx * SURFEL_BUCKET_SIZE + lane;
	const int instance_idx = range.x + tile_primitive_idx;
	const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

	const uint32_t n_pixels = (uint32_t)W * (uint32_t)H;
	const float3 background = make_float3(bg_color[0], bg_color[1], bg_color[2]);

	// Load this lane's primitive data.
	uint32_t global_id = 0;
	float2 xy = {0.0f, 0.0f};
	float3 Tu = {0.0f, 0.0f, 0.0f}, Tv = {0.0f, 0.0f, 0.0f}, Tw = {0.0f, 0.0f, 0.0f};
	float normal[3] = {0.0f, 0.0f, 0.0f};
	float opacity = 0.0f;
	float color[3] = {0.0f, 0.0f, 0.0f};
	float color_grad_factor[3] = {0.0f, 0.0f, 0.0f};
	if (valid_primitive) {
		global_id = point_list[instance_idx];
		xy = points_xy_image[global_id];
		Tu = {transMats[9 * global_id + 0], transMats[9 * global_id + 1], transMats[9 * global_id + 2]};
		Tv = {transMats[9 * global_id + 3], transMats[9 * global_id + 4], transMats[9 * global_id + 5]};
		Tw = {transMats[9 * global_id + 6], transMats[9 * global_id + 7], transMats[9 * global_id + 8]};
		const float4 nor_o = normal_opacity[global_id];
		normal[0] = nor_o.x; normal[1] = nor_o.y; normal[2] = nor_o.z; opacity = nor_o.w;
		for (int ch = 0; ch < 3; ch++) {
			const float cu = colors[global_id * 3 + ch];
			color[ch] = fmaxf(cu, 0.0f);
			color_grad_factor[ch] = (cu >= 0.0f) ? 1.0f : 0.0f;
		}
	}

	// Per-primitive gradient accumulators (summed over all pixels of the tile).
	float dL_dTu[3] = {0.0f, 0.0f, 0.0f};
	float dL_dTv[3] = {0.0f, 0.0f, 0.0f};
	float dL_dTw[3] = {0.0f, 0.0f, 0.0f};
	float2 dL_dmean2d_acc = {0.0f, 0.0f};
	float dL_dnormal_acc[3] = {0.0f, 0.0f, 0.0f};
	float dL_dopacity_acc = 0.0f;
	float dL_dcolor_acc[3] = {0.0f, 0.0f, 0.0f};

	const uint2 tile_coords = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 start_pixel = {tile_coords.x * BLOCK_X, tile_coords.y * BLOCK_Y};

	// Per-bucket checkpoint slice (running accumulators BEFORE this bucket).
	const float4* ckpt_color_T = bucket_color_T + (size_t)bucket_idx * BLOCK_SIZE;
	const float4* ckpt_aux_dn = bucket_aux_dn + (size_t)bucket_idx * BLOCK_SIZE;
	const float2* ckpt_aux_m = bucket_aux_m + (size_t)bucket_idx * BLOCK_SIZE;

	// Shared per-pixel state for the current group of 32 pixels.
	__shared__ float sh_color_after[3][32];
	__shared__ float sh_T[32];
	__shared__ float sh_D_after[32];
	__shared__ float sh_N_after[3][32];
	__shared__ float sh_alpha_after[32];
	__shared__ float sh_G_after[32];
	__shared__ float sh_dL_dpix[3][32];
	__shared__ float sh_dL_ddepth[32];
	__shared__ float sh_dL_daccum[32];
	__shared__ float sh_dL_dnormal[3][32];
	__shared__ float sh_dL_dreg[32];
	__shared__ float sh_dL_dmedian[32];
	__shared__ float sh_grad_alpha_common[32];
	__shared__ float sh_final_A[32];
	__shared__ float sh_final_M1[32];
	__shared__ float sh_final_M2[32];
	__shared__ uint32_t sh_last_contrib[32];
	__shared__ uint32_t sh_median_idx[32];

	// Carried per-pixel registers (the pixel this lane handles at the current step).
	float r_color_after[3] = {0.0f, 0.0f, 0.0f}, r_T = 1.0f, r_D_after = 0.0f;
	float r_N_after[3] = {0.0f, 0.0f, 0.0f}, r_alpha_after = 0.0f, r_G_after = 0.0f;
	float r_dL_dpix[3] = {0.0f, 0.0f, 0.0f}, r_dL_ddepth = 0.0f, r_dL_daccum = 0.0f;
	float r_dL_dnormal[3] = {0.0f, 0.0f, 0.0f}, r_dL_dreg = 0.0f, r_dL_dmedian = 0.0f, r_grad_alpha_common = 0.0f;
	float r_final_A = 0.0f, r_final_M1 = 0.0f, r_final_M2 = 0.0f;
	uint32_t r_last_contrib = 0, r_median_idx = 0;

	for (int i = 0; i < BLOCK_SIZE + 31; ++i)
	{
		if (i % 32 == 0)
		{
			const int local_idx = i + lane;
			if (local_idx < BLOCK_SIZE)
			{
				const uint2 pc = {start_pixel.x + local_idx % BLOCK_X, start_pixel.y + local_idx / BLOCK_X};
				const bool pix_inside = pc.x < (uint32_t)W && pc.y < (uint32_t)H;
				const uint32_t pid = W * pc.y + pc.x;

				float Tf = 0.0f, M1f = 0.0f, M2f = 0.0f;
				float3 cfinal = {0.0f, 0.0f, 0.0f};
				float Dfinal = 0.0f; float Nfinal[3] = {0.0f, 0.0f, 0.0f};
				float dpix[3] = {0.0f, 0.0f, 0.0f};
				float dd = 0.0f, da = 0.0f, dr = 0.0f, dn[3] = {0.0f, 0.0f, 0.0f}, dm = 0.0f;
				uint32_t lastc = 0, medi = 0;
				float4 cT = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
				float4 adn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				float2 amm = make_float2(0.0f, 0.0f);
				if (pix_inside)
				{
					Tf = final_T[pid]; M1f = final_M1[pid]; M2f = final_M2[pid];
					cfinal = make_float3(
						out_color[0 * n_pixels + pid] - Tf * background.x,
						out_color[1 * n_pixels + pid] - Tf * background.y,
						out_color[2 * n_pixels + pid] - Tf * background.z);
					Dfinal = out_others[DEPTH_OFFSET * n_pixels + pid];
					for (int ch = 0; ch < 3; ch++) Nfinal[ch] = out_others[(NORMAL_OFFSET + ch) * n_pixels + pid];
					for (int ch = 0; ch < 3; ch++) dpix[ch] = dL_dpixels[ch * n_pixels + pid];
					dd = dL_depths[DEPTH_OFFSET * n_pixels + pid];
					da = dL_depths[ALPHA_OFFSET * n_pixels + pid];
					dr = dL_depths[DISTORTION_OFFSET * n_pixels + pid];
					for (int ch = 0; ch < 3; ch++) dn[ch] = dL_depths[(NORMAL_OFFSET + ch) * n_pixels + pid];
					dm = dL_depths[MIDDEPTH_OFFSET * n_pixels + pid];
					lastc = n_contrib[pid]; medi = median_contrib[pid];
					cT = ckpt_color_T[local_idx];
					adn = ckpt_aux_dn[local_idx];
					amm = ckpt_aux_m[local_idx];
				}
				const float T_pre = cT.w;
				const float A_pre = 1.0f - T_pre;
				const float final_A = 1.0f - Tf;
				const float G_total = 2.0f * (final_A * M2f - M1f * M1f);
				const float prefix_before = final_A * amm.y + M2f * A_pre - 2.0f * M1f * amm.x;

				sh_color_after[0][lane] = cfinal.x - cT.x;
				sh_color_after[1][lane] = cfinal.y - cT.y;
				sh_color_after[2][lane] = cfinal.z - cT.z;
				sh_T[lane] = T_pre;
				sh_D_after[lane] = Dfinal - adn.x;
				sh_N_after[0][lane] = Nfinal[0] - adn.y;
				sh_N_after[1][lane] = Nfinal[1] - adn.z;
				sh_N_after[2][lane] = Nfinal[2] - adn.w;
				sh_alpha_after[lane] = T_pre - Tf;
				sh_G_after[lane] = G_total - prefix_before;
				for (int ch = 0; ch < 3; ch++) sh_dL_dpix[ch][lane] = dpix[ch];
				sh_dL_ddepth[lane] = dd;
				sh_dL_daccum[lane] = da;
				for (int ch = 0; ch < 3; ch++) sh_dL_dnormal[ch][lane] = dn[ch];
				sh_dL_dreg[lane] = dr;
				sh_dL_dmedian[lane] = dm;
				sh_grad_alpha_common[lane] = Tf * -(dpix[0] * background.x + dpix[1] * background.y + dpix[2] * background.z);
				sh_final_A[lane] = final_A;
				sh_final_M1[lane] = M1f;
				sh_final_M2[lane] = M2f;
				sh_last_contrib[lane] = lastc;
				sh_median_idx[lane] = medi;
			}
			warp.sync();
		}

		if (i > 0)
		{
			for (int ch = 0; ch < 3; ch++) {
				r_color_after[ch] = warp.shfl_up(r_color_after[ch], 1);
				r_N_after[ch] = warp.shfl_up(r_N_after[ch], 1);
				r_dL_dpix[ch] = warp.shfl_up(r_dL_dpix[ch], 1);
				r_dL_dnormal[ch] = warp.shfl_up(r_dL_dnormal[ch], 1);
			}
			r_T = warp.shfl_up(r_T, 1);
			r_D_after = warp.shfl_up(r_D_after, 1);
			r_alpha_after = warp.shfl_up(r_alpha_after, 1);
			r_G_after = warp.shfl_up(r_G_after, 1);
			r_dL_ddepth = warp.shfl_up(r_dL_ddepth, 1);
			r_dL_daccum = warp.shfl_up(r_dL_daccum, 1);
			r_dL_dreg = warp.shfl_up(r_dL_dreg, 1);
			r_dL_dmedian = warp.shfl_up(r_dL_dmedian, 1);
			r_grad_alpha_common = warp.shfl_up(r_grad_alpha_common, 1);
			r_final_A = warp.shfl_up(r_final_A, 1);
			r_final_M1 = warp.shfl_up(r_final_M1, 1);
			r_final_M2 = warp.shfl_up(r_final_M2, 1);
			r_last_contrib = warp.shfl_up(r_last_contrib, 1);
			r_median_idx = warp.shfl_up(r_median_idx, 1);
		}

		const int idx = i - (int)lane;
		const int idx_clamped = (idx >= 0 && idx < BLOCK_SIZE) ? idx : 0;
		const uint2 pc = {start_pixel.x + (uint32_t)(idx_clamped % BLOCK_X),
						  start_pixel.y + (uint32_t)(idx_clamped / BLOCK_X)};
		const bool valid_pixel = idx >= 0 && idx < BLOCK_SIZE && pc.x < (uint32_t)W && pc.y < (uint32_t)H;

		// Leader thread reads freshly loaded shared state into registers.
		if (lane == 0 && idx >= 0 && idx < BLOCK_SIZE)
		{
			const int s = i % 32;
			for (int ch = 0; ch < 3; ch++) {
				r_color_after[ch] = sh_color_after[ch][s];
				r_N_after[ch] = sh_N_after[ch][s];
				r_dL_dpix[ch] = sh_dL_dpix[ch][s];
				r_dL_dnormal[ch] = sh_dL_dnormal[ch][s];
			}
			r_T = sh_T[s];
			r_D_after = sh_D_after[s];
			r_alpha_after = sh_alpha_after[s];
			r_G_after = sh_G_after[s];
			r_dL_ddepth = sh_dL_ddepth[s];
			r_dL_daccum = sh_dL_daccum[s];
			r_dL_dreg = sh_dL_dreg[s];
			r_dL_dmedian = sh_dL_dmedian[s];
			r_grad_alpha_common = sh_grad_alpha_common[s];
			r_final_A = sh_final_A[s];
			r_final_M1 = sh_final_M1[s];
			r_final_M2 = sh_final_M2[s];
			r_last_contrib = sh_last_contrib[s];
			r_median_idx = sh_median_idx[s];
		}

		if (!valid_primitive || !valid_pixel || (uint32_t)tile_primitive_idx >= r_last_contrib)
			continue;

		const float2 pixf = {(float)pc.x, (float)pc.y};

		// Recompute the surfel ray-splat intersection exactly as in the forward.
		float3 kk = {pixf.x * Tw.x - Tu.x, pixf.x * Tw.y - Tu.y, pixf.x * Tw.z - Tu.z};
		float3 ll = {pixf.y * Tw.x - Tv.x, pixf.y * Tw.y - Tv.y, pixf.y * Tw.z - Tv.z};
		float3 pp = cross(kk, ll);
		if (pp.z == 0.0f) continue;
		float2 ss = {pp.x / pp.z, pp.y / pp.z};
		float rho3d = ss.x * ss.x + ss.y * ss.y;
		float2 dd2 = {xy.x - pixf.x, xy.y - pixf.y};
		float rho2d = FilterInvSquare * (dd2.x * dd2.x + dd2.y * dd2.y);
		float rho = min(rho3d, rho2d);
		float depth = ss.x * Tw.x + ss.y * Tw.y + Tw.z;
		if (depth < near_n) continue;
		float power = -0.5f * rho;
		if (power > 0.0f) continue;
		float G = exp(power);
		float alpha = min(0.99f, opacity * G);
		if (alpha < 1.0f / 255.0f) continue;

		const float T_i = r_T;
		const float one_minus_alpha = 1.0f - alpha;
		const float omar = 1.0f / fmaxf(one_minus_alpha, 1e-6f);
		const float w = alpha * T_i;
		const float m = far_n / (far_n - near_n) * (1.0f - near_n / depth);
		const float g_i = m * m * r_final_A + r_final_M2 - 2.0f * m * r_final_M1;

		// Update strictly-after suffix sums (subtract current contribution).
		for (int ch = 0; ch < 3; ch++) r_color_after[ch] -= w * color[ch];
		r_D_after -= w * depth;
		for (int ch = 0; ch < 3; ch++) r_N_after[ch] -= w * normal[ch];
		r_alpha_after -= w;
		r_G_after -= w * g_i;

		// dL/dalpha accumulation (FastGS convention: includes the T_i factor inline).
		float dL_dalpha = 0.0f;
		for (int ch = 0; ch < 3; ch++)
			dL_dalpha += (T_i * color[ch] - r_color_after[ch] * omar) * r_dL_dpix[ch];
		dL_dalpha += r_grad_alpha_common * omar;                                  // background
		dL_dalpha += (T_i * depth - r_D_after * omar) * r_dL_ddepth;              // expected depth
		dL_dalpha += (T_i - r_alpha_after * omar) * r_dL_daccum;                  // accumulated alpha
		for (int ch = 0; ch < 3; ch++)
			dL_dalpha += (T_i * normal[ch] - r_N_after[ch] * omar) * r_dL_dnormal[ch]; // normal
		dL_dalpha += (T_i * g_i - r_G_after * omar) * r_dL_dreg;                  // distortion

		// Direct color gradient.
		for (int ch = 0; ch < 3; ch++)
			dL_dcolor_acc[ch] += w * r_dL_dpix[ch] * color_grad_factor[ch];

		// Depth (ray-splat) gradient.
		float dL_dz = 0.0f;
		if (r_median_idx != 0 && (uint32_t)(tile_primitive_idx + 1) == r_median_idx)
			dL_dz += r_dL_dmedian;
		const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * depth * depth);
		const float dL_dmd = 2.0f * w * (m * r_final_A - r_final_M1) * r_dL_dreg;
		dL_dz += dL_dmd * dmd_dd;
		dL_dz += w * r_dL_ddepth;

		// Direct normal gradient.
		for (int ch = 0; ch < 3; ch++)
			dL_dnormal_acc[ch] += w * r_dL_dnormal[ch];

		const float dL_dG = opacity * dL_dalpha;
		dL_dopacity_acc += G * dL_dalpha;

		if (rho3d <= rho2d) {
			const float2 dL_ds = {dL_dG * -G * ss.x + dL_dz * Tw.x, dL_dG * -G * ss.y + dL_dz * Tw.y};
			const float3 dz_dTw = {ss.x, ss.y, 1.0f};
			const float dsx_pz = dL_ds.x / pp.z;
			const float dsy_pz = dL_ds.y / pp.z;
			const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * ss.x + dsy_pz * ss.y)};
			const float3 dL_dk = cross(ll, dL_dp);
			const float3 dL_dl = cross(dL_dp, kk);
			dL_dTu[0] += -dL_dk.x; dL_dTu[1] += -dL_dk.y; dL_dTu[2] += -dL_dk.z;
			dL_dTv[0] += -dL_dl.x; dL_dTv[1] += -dL_dl.y; dL_dTv[2] += -dL_dl.z;
			dL_dTw[0] += pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x;
			dL_dTw[1] += pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y;
			dL_dTw[2] += pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z;
		} else {
			const float dG_ddelx = -G * FilterInvSquare * dd2.x;
			const float dG_ddely = -G * FilterInvSquare * dd2.y;
			dL_dmean2d_acc.x += dL_dG * dG_ddelx;
			dL_dmean2d_acc.y += dL_dG * dG_ddely;
			dL_dTw[0] += ss.x * dL_dz;
			dL_dTw[1] += ss.y * dL_dz;
			dL_dTw[2] += dL_dz;
		}

		// Advance transmittance for the next lane handling this pixel.
		r_T *= one_minus_alpha;
	}

	if (valid_primitive)
	{
		atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu[0]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu[1]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu[2]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv[0]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv[1]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv[2]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw[0]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw[1]);
		atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw[2]);
		atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2d_acc.x);
		atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2d_acc.y);
		for (int ch = 0; ch < 3; ch++) atomicAdd(&dL_dnormal3D[global_id * 3 + ch], dL_dnormal_acc[ch]);
		atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
		for (int ch = 0; ch < 3; ch++) atomicAdd(&dL_dcolors[global_id * 3 + ch], dL_dcolor_acc[ch]);
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
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

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);
}

void BACKWARD::render(
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
	float* dL_dcolors)
{
	if (n_buckets == 0) return;
	renderCUDA<NUM_CHANNELS> << <n_buckets, 32 >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		out_color,
		out_others,
		final_T,
		final_M1,
		final_M2,
		n_contrib,
		median_contrib,
		max_contrib,
		tile_bucket_offsets,
		bucket_tile_index,
		bucket_color_T,
		bucket_aux_dn,
		bucket_aux_m,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors
		);
}
