#pragma once

#include <smith/csrc/jit/python/pybind.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::jit {
// Initialize Python bindings for JIT to_<backend> functions.
void initJitBackendBindings(PyObject* module);
} // namespace smith::jit
