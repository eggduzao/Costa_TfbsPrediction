#pragma once
#include <smith/csrc/python_headers.h>
// C2039 MSVC
#include <pybind11/complex.h>
#include <smith/csrc/utils/pybind.h>

#include <Python.h>
// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace smith::dynamo {
PyObject* smith_c_dynamo_utils_init();
} // namespace smith::dynamo
