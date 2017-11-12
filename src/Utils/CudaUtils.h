#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_


#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
#define cudaError_t int
#endif

class CudaUtils
{
public:

	static void checkError(cudaError_t code, const std::string& message);
};
#endif
