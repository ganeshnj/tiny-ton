add_library(TinyTonConversion
  TinyTonToGPU.cpp
)

target_include_directories(TinyTonConversion PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(TinyTonConversion PUBLIC
  TinyTonDialect
  MLIRGPUDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRIR
  MLIRSupport
)
