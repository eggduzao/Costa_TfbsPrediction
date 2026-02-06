#pragma once
#include <smith/csrc/utils/python_stub.h>

// see [Note: Compiled Autograd]
namespace smith::dynamo::autograd {
PyObject* smith_c_dynamo_compiled_autograd_init();
} // namespace smith::dynamo::autograd
