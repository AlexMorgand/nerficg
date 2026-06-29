/*
 * Faster2DGS native surfel rasterizer (NeRFICG).
 * Bucket-checkpoint orchestration from FastGS; surfel forward/backward implements
 * the 2D Gaussian Splatting perspective-disk formulation.
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// FastGS-style bucket checkpointing: one bucket = 32 consecutive (depth-sorted)
// primitives within a tile. The bucket-parallel backward launches <<<n_buckets, 32>>>,
// decoupling backward parallelism from the per-tile serial loop of the original 2DGS.
#define SURFEL_BUCKET_SIZE 32

#endif