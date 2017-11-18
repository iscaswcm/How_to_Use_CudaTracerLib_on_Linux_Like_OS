#include "core/precompiled.h"

#include "tinyformat/tinyformat.h"

#include "core/cudautils.h"

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
	(void)code;
	(void)message;

#ifdef USE_CUDA
	if(code != cudaSuccess)
		throw std::runtime_error(tfm::format("Cuda error: %s: %s", message, cudaGetErrorString(code)));
#endif
}
