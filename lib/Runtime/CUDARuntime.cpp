#include "tiny-ton/Runtime/CUDARuntime.h"

#include <cuda.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *msg = nullptr;                                               \
      cuGetErrorString(err, &msg);                                             \
      throw std::runtime_error(std::string(#call " failed: ") +               \
                               (msg ? msg : "unknown error"));                 \
    }                                                                          \
  } while (0)

namespace tinyton {

CUDARuntime::CUDARuntime() {
  CUDA_CHECK(cuInit(0));

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  CUcontext cuCtx;
  CUDA_CHECK(cuCtxCreate(&cuCtx, 0, device));
  ctx_ = cuCtx;
}

CUDARuntime::~CUDARuntime() {
  if (ctx_) {
    cuCtxDestroy(static_cast<CUcontext>(ctx_));
  }
}

void *CUDARuntime::alloc(size_t bytes) {
  CUdeviceptr ptr;
  CUDA_CHECK(cuMemAlloc(&ptr, bytes));
  return reinterpret_cast<void *>(ptr);
}

void CUDARuntime::free(void *devPtr) {
  CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(devPtr)));
}

void CUDARuntime::copyToDevice(void *dst, const void *src, size_t bytes) {
  CUDA_CHECK(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, bytes));
}

void CUDARuntime::copyFromDevice(void *dst, const void *src, size_t bytes) {
  CUDA_CHECK(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), bytes));
}

void CUDARuntime::launch(const std::string &ptx,
                         const std::string &kernelName, int gridX, int blockX,
                         const std::vector<void *> &kernelArgs) {
  CUmodule cuModule;
  CUDA_CHECK(cuModuleLoadData(&cuModule, ptx.c_str()));

  CUfunction cuFunc;
  CUDA_CHECK(cuModuleGetFunction(&cuFunc, cuModule, kernelName.c_str()));

  std::vector<void *> argsCopy(kernelArgs);
  std::vector<void *> argPtrs(argsCopy.size());
  for (size_t i = 0; i < argsCopy.size(); ++i)
    argPtrs[i] = &argsCopy[i];

  CUDA_CHECK(cuLaunchKernel(cuFunc, gridX, 1, 1, blockX, 1, 1, 0, nullptr,
                            argPtrs.data(), nullptr));

  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuModuleUnload(cuModule));
}

bool CUDARuntime::isAvailable() {
  if (cuInit(0) != CUDA_SUCCESS)
    return false;
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS)
    return false;
  return count > 0;
}

} // namespace tinyton
