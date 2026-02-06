#pragma once

// Instantiates smith._C._LegacyVariableBase, which defines the Python
// constructor (__new__) for smith.autograd.Variable.

#include <smith/csrc/python_headers.h>

namespace smith::autograd {

void init_legacy_variable(PyObject* module);

}
