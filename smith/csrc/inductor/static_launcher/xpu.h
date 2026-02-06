#pragma once
#if defined(USE_XPU)
#include <smith/csrc/python_headers.h>

bool StaticXpuLauncher_init(PyObject* module);
#endif
