set(RUNTIME_SOURCES Simulator.cpp)

if(TTN_ENABLE_CUDA)
  list(APPEND RUNTIME_SOURCES CUDARuntime.cpp)
endif()

add_library(TinyTonRuntime ${RUNTIME_SOURCES})

target_link_libraries(TinyTonRuntime
  PUBLIC
  TinyTonCompiler
)

if(TTN_ENABLE_CUDA)
  target_link_libraries(TinyTonRuntime PUBLIC CUDA::cuda_driver)
endif()
