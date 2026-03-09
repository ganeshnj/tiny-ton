#include "tiny-ton/Runtime/Simulator.h"

namespace tinyton {

struct SimulatedGPU::Impl {
  std::vector<uint16_t> program;
  std::vector<uint8_t> dataMemory;
};

SimulatedGPU::SimulatedGPU() : impl_(std::make_unique<Impl>()) {}
SimulatedGPU::~SimulatedGPU() = default;

void SimulatedGPU::loadProgram(const std::vector<uint16_t> &instructions) {
  impl_->program = instructions;
}

void SimulatedGPU::loadDataMemory(const std::vector<uint8_t> &data) {
  impl_->dataMemory = data;
}

void SimulatedGPU::run(int numBlocks, int threadsPerBlock) {
  // TODO: implement cycle-accurate (or functional) GPU simulation
}

std::vector<uint8_t> SimulatedGPU::readDataMemory() const {
  return impl_->dataMemory;
}

} // namespace tinyton
