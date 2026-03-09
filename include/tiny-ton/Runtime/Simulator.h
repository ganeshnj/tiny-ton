#pragma once

#include <cstdint>
#include <vector>

namespace tinyton {

class SimulatedGPU {
public:
  SimulatedGPU();
  ~SimulatedGPU();

  void loadProgram(const std::vector<uint16_t> &instructions);
  void loadDataMemory(const std::vector<uint8_t> &data);
  void run(int numBlocks, int threadsPerBlock);
  std::vector<uint8_t> readDataMemory() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tinyton
