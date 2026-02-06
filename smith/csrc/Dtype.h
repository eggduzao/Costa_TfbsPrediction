#pragma once

#include <c10/core/ScalarType.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/python_headers.h>

constexpr int DTYPE_NAME_LEN = 64;

struct SMITH_API THPDtype {
  PyObject_HEAD
  at::ScalarType scalar_type;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[DTYPE_NAME_LEN + 1];
};

SMITH_API extern PyTypeObject THPDtypeType;

inline bool THPDtype_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPDtypeType;
}

inline bool THPPythonScalarType_Check(PyObject* obj) {
  return obj == (PyObject*)(&PyFloat_Type) ||
      obj == (PyObject*)(&PyComplex_Type) || obj == (PyObject*)(&PyBool_Type) ||
      obj == (PyObject*)(&PyLong_Type);
}

SMITH_API PyObject* THPDtype_New(
    at::ScalarType scalar_type,
    const std::string& name);

void THPDtype_init(PyObject* module);
