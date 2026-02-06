#pragma once

#include <smith/csrc/jit/python/pybind.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::jit {
// Initialize Python bindings for Tensor Expressions
void initTensorExprBindings(PyObject* module);
} // namespace smith::jit
