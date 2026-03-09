#include "tiny-ton/Runtime/Runtime.h"

namespace tinyton {

struct Runtime::Impl {
  // TODO: memory management state
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {}
Runtime::~Runtime() = default;

void Runtime::launch(const CompiledKernel &kernel, const LaunchParams &params,
                     const std::vector<void *> &args) {
  // TODO: implement — set up memory, dispatch to simulator or real GPU
}

} // namespace tinyton
