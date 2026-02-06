#pragma once
#if defined(USE_CUDA)
#include <smith/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#include <smith/csrc/python_headers.h>

bool StaticCudaLauncher_init(PyObject* module);
#endif
