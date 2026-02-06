#pragma once

#include <smith/csrc/utils/python_compat.h>
namespace smith::autograd {

PyMethodDef* get_nested_functions_manual();

void initNestedFunctions(PyObject* module);

} // namespace smith::autograd
