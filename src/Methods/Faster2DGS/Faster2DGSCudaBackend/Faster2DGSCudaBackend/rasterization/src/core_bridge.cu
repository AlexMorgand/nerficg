// Embed FastGS rasterization kernels under namespace faster2dgs_core (token remap).

#define faster_gs faster2dgs_core

#include "../../../../../FasterGS/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/src/forward.cu"
#include "../../../../../FasterGS/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/src/backward.cu"
#include "../../../../../FasterGS/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/src/inference.cu"
