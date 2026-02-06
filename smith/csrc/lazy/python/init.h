#pragma once
#include <pybind11/pybind11.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::lazy {

SMITH_PYTHON_API void initLazyBindings(PyObject* module);

} // namespace smith::lazy
