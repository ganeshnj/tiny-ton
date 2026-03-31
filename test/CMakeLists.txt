set(COMMON_TEST_LIBS
  TinyTonIR
  TinyTonCompiler
  TinyTonRuntime
  TinyTonDialect
  MLIRIR
  MLIRSupport
)

# Existing test
add_llvm_executable(test_add_e2e test_add_e2e.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_add_e2e PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME add_e2e COMMAND test_add_e2e)

# E2E simulator tests
add_llvm_executable(test_arith_e2e test_arith_e2e.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_arith_e2e PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME arith_e2e COMMAND test_arith_e2e)

add_llvm_executable(test_math_e2e test_math_e2e.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_math_e2e PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME math_e2e COMMAND test_math_e2e)

add_llvm_executable(test_masked_load_store_e2e test_masked_load_store_e2e.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_masked_load_store_e2e PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME masked_load_store_e2e COMMAND test_masked_load_store_e2e)

add_llvm_executable(test_reduce_e2e test_reduce_e2e.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_reduce_e2e PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME reduce_e2e COMMAND test_reduce_e2e)

# Unit tests
add_llvm_executable(test_register_alloc test_register_alloc.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_register_alloc PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME register_alloc COMMAND test_register_alloc)

add_llvm_executable(test_codegen test_codegen.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_codegen PRIVATE ${COMMON_TEST_LIBS})
add_test(NAME codegen COMMAND test_codegen)

# GPU lowering test
add_llvm_executable(test_gpu_lowering test_gpu_lowering.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(test_gpu_lowering PRIVATE
  ${COMMON_TEST_LIBS}
  TinyTonConversion
)
add_test(NAME gpu_lowering COMMAND test_gpu_lowering)
