#pragma once

// C2039 MSVC
#include <pybind11/complex.h>
#include <smith/csrc/utils/pybind.h>

#include <Python.h>

namespace smith::dynamo {
void initDynamoBindings(PyObject* smith);
}
