#pragma once
#include <smith/csrc/utils/pythoncapi_compat.h>

namespace smith::autograd {

void initFFTFunctions(PyObject* module);

}
