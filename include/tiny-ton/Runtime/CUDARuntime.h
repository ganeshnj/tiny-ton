#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tinyton {

class CUDARuntime {
public:
  CUDARuntime();
  ~CUDARuntime();

  CUDARuntime(const CUDARuntime &) = delete;
  CUDARuntime &operator=(const CUDARuntime &) = delete;

  void *alloc(size_t bytes);
  void free(void *devPtr);

  void copyToDevice(void *dst, const void *src, size_t bytes);
  void copyFromDevice(void *dst, const void *src, size_t bytes);

  void launch(const std::string &ptx, const std::string &kernelName, int gridX,
              int blockX, const std::vector<void *> &kernelArgs);

  static bool isAvailable();

private:
  void *ctx_ = nullptr;
};

} // namespace tinyton
