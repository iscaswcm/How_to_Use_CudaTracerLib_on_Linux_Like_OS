#ifndef _CORE_COMMON_H_
#define CUDATRACERLIB_VERSION "0.1.0"
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE 
#endif
#endif
