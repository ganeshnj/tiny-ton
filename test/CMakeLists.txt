add_llvm_executable(test_add_e2e
  test_add_e2e.cpp
)
target_link_libraries(test_add_e2e
  PRIVATE
  TinyTonIR
  TinyTonCompiler
  TinyTonRuntime
  TinyTonDialect
  MLIRIR
  MLIRSupport
)

add_test(NAME add_e2e COMMAND test_add_e2e)
