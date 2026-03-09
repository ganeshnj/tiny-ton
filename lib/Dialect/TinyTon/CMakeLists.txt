add_mlir_dialect_library(TinyTonDialect
  TinyTonDialect.cpp
  TinyTonOps.cpp

  DEPENDS
  TinyTonOpsIncGen
)
