add_library(TinyTonRuntime
  Simulator.cpp
)
target_link_libraries(TinyTonRuntime
  PUBLIC
  TinyTonCompiler
)
