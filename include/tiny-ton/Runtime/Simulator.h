#pragma once

#include <cstdint>
#include <vector>

namespace tinyton {

class SimulatedGPU {
public:
  explicit SimulatedGPU(int memWords = 4096);
  ~SimulatedGPU();

  void loadProgram(const std::vector<uint16_t> &instructions);
  void setArgs(const std::vector<int32_t> &args);
  void writeMemory(int addr, const std::vector<int32_t> &data);
  std::vector<int32_t> readMemory(int addr, int count) const;
  void run(int numBlocks, int threadsPerBlock);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tinyton
