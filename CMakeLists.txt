cmake_minimum_required(VERSION 3.20)
project(tiny-ton LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LLVM / MLIR -----------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# --- CUDA (optional) --------------------------------------------------------
option(TTN_ENABLE_CUDA "Build CUDA/PTX backend for real GPU execution" OFF)
if(TTN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA toolkit found: ${CUDAToolkit_LIBRARY_DIR}")
  add_compile_definitions(TTN_ENABLE_CUDA)
endif()

# --- Sub-projects -----------------------------------------------------------
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# --- pybind11 bindings (optional) -------------------------------------------
option(TTN_ENABLE_PYTHON "Build Python bindings via pybind11" ON)
if(TTN_ENABLE_PYTHON)
  add_subdirectory(bindings)
endif()

# --- Testing (optional) -----------------------------------------------------
option(TTN_ENABLE_TESTS "Enable tests" ON)
if(TTN_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
