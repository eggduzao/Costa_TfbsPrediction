#pragma once
#include <smith/csrc/utils/pythoncapi_compat.h>

namespace smith::autograd {

void initSparseFunctions(PyObject* module);

}
