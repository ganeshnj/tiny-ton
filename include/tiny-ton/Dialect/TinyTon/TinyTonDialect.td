// TinyTon dialect definition.

#ifndef TINYTON_DIALECT
#define TINYTON_DIALECT

include "mlir/IR/OpBase.td"

def TinyTon_Dialect : Dialect {
  let name = "tinyton";
  let summary = "A Triton-inspired dialect for GPU kernel compilation.";
  let description = [{
    The TinyTon dialect provides operations for expressing GPU kernels that
    can be compiled to run on simulated or real GPU hardware. It models
    block-level parallelism with explicit memory operations.
  }];
  let cppNamespace = "::tinyton";
}

#endif // TINYTON_DIALECT
