find_package(pybind11 REQUIRED)

pybind11_add_module(_tiny_ton_core python_bindings.cpp)

target_link_libraries(_tiny_ton_core
  PRIVATE
  TinyTonIR
  TinyTonCompiler
  TinyTonRuntime
)
