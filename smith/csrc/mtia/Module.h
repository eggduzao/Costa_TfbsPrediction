#pragma once

#include <smith/csrc/python_headers.h>

namespace smith::mtia {

// PyMethodDef* python_functions();
void initModule(PyObject* module);

} // namespace smith::mtia
