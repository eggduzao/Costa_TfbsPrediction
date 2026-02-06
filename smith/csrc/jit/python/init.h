#pragma once

#include <smith/csrc/utils/pybind.h>

namespace smith::jit {

void initJITBindings(PyObject* module);

} // namespace smith::jit
