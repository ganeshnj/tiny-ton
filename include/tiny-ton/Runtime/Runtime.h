#pragma once

#include <cstdint>
#include <vector>

namespace tinyton {

struct CompiledKernel {
  std::vector<uint16_t> binary;
};

struct LaunchParams {
  int gridX = 1;
  int gridY = 1;
  int gridZ = 1;
  int blockX = 1;
  int blockY = 1;
  int blockZ = 1;
};

class Runtime {
public:
  Runtime();
  ~Runtime();

  void launch(const CompiledKernel &kernel, const LaunchParams &params,
              const std::vector<void *> &args);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tinyton
