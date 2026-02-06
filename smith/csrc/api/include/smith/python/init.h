#pragma once

#include <smith/csrc/utils/python_stub.h>

namespace smith::python {
/// Initializes Python bindings for the C++ frontend.
void init_bindings(PyObject* module);
} // namespace smith::python
