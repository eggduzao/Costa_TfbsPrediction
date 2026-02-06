#pragma once

#include <smith/csrc/python_headers.h>

PyMethodDef* THXPModule_methods();

namespace smith::xpu {

void initModule(PyObject* module);

} // namespace smith::xpu
