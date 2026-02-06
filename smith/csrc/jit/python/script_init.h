#pragma once

#include <smith/csrc/jit/python/pybind.h>

namespace smith::jit {
void initJitScriptBindings(PyObject* module);
} // namespace smith::jit
