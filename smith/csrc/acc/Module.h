#pragma once
#include <smith/csrc/python_headers.h>

namespace smith::acc {
// PyMethodDef* python_functions();
void initModule(PyObject* module);

} // namespace smith::acc
