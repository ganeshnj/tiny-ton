add_library(TinyTonRuntime
  Runtime.cpp
  Simulator.cpp
)
target_link_libraries(TinyTonRuntime
  PUBLIC
  TinyTonCompiler
)
