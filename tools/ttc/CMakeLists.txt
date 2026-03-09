add_llvm_executable(ttc
  ttc.cpp
)
target_link_libraries(ttc
  PRIVATE
  TinyTonCompiler
  TinyTonRuntime
)
