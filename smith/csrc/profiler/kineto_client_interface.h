#pragma once

#include <smith/csrc/jit/runtime/interpreter.h>
#include <smith/csrc/profiler/unwind/unwind.h>

namespace smith {

// declare global_kineto_init for libsmith_cpu.so to call
SMITH_API void global_kineto_init();

} // namespace smith
