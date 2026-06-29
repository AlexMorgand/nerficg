/*
 * Faster2DGS native surfel rasterizer (NeRFICG).
 * Bucket-checkpoint orchestration from FastGS; surfel forward/backward implements
 * the 2D Gaussian Splatting perspective-disk formulation.
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

// Pixels per tile (BLOCK_X * BLOCK_Y). Kept as a literal here so this header does
// not need config.h, which would define NUM_CHANNELS before cub is included in the
// translation unit and break cub's histogram templates.
#define SURFEL_TILE_PIXELS 256

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// Per-primitive state (preprocess outputs + sorting scratch). Unchanged from 2DGS.
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* transMat;
		float4* normal_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	// Per-instance (duplicated tile/primitive) state + per-tile and per-pixel
	// forward checkpoints needed by the bucket-parallel backward.
	struct BinningState
	{
		// tile|depth sorting of duplicated primitives
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;
		// per-tile state
		uint2* ranges;            // [n_tiles] instance range per tile
		uint32_t* tile_n_buckets; // [n_tiles] number of 32-wide buckets per tile
		uint32_t* tile_bucket_offsets; // [n_tiles] inclusive-scan global bucket prefix
		uint32_t* max_contrib;    // [n_tiles] max last-contributor per tile (backward guard)
		size_t bucket_scan_size;
		char* bucket_scan_space;
		// per-pixel forward finals (read in backward to reconstruct aux suffix sums)
		float* final_T;           // [N] final transmittance
		float* final_M1;          // [N] distortion accumulator sum(w*m)
		float* final_M2;          // [N] distortion accumulator sum(w*m*m)
		uint32_t* n_contrib;      // [N] last contributor index (count of used)
		uint32_t* median_contrib; // [N] 0-based index of median-depth splat (+1; 0 = none)

		static BinningState fromChunk(char*& chunk, size_t num_rendered, int n_tiles, int N);
	};

	// Per-bucket forward checkpoints (one block per bucket in backward).
	// Each bucket stores, for all 256 pixels of its tile, the running accumulators
	// BEFORE the bucket's 32 primitives are blended.
	struct BucketState
	{
		uint32_t* tile_index;     // [n_buckets] owning tile
		float4* color_T;          // [n_buckets * 256] (color.rgb, transmittance)
		float4* aux_dn;           // [n_buckets * 256] (expected_depth_sum D, normal.xyz N)
		float2* aux_m;            // [n_buckets * 256] (distortion M1, M2)

		static BucketState fromChunk(char*& chunk, size_t n_buckets);
	};

	template <typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	inline size_t required_binning(size_t num_rendered, int n_tiles, int N)
	{
		char* size = nullptr;
		BinningState::fromChunk(size, num_rendered, n_tiles, N);
		return ((size_t)size) + 128;
	}

	inline size_t required_buckets(size_t n_buckets)
	{
		char* size = nullptr;
		BucketState::fromChunk(size, n_buckets);
		return ((size_t)size) + 128;
	}
};
