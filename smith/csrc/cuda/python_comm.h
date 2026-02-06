#pragma once

#include <smith/csrc/utils/pythoncapi_compat.h>
namespace smith::cuda::python {

void initCommMethods(PyObject* module);

} // namespace smith::cuda::python
