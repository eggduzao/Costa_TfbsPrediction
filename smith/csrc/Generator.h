#pragma once

#include <ATen/core/Generator.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPGenerator {
  PyObject_HEAD
  at::Generator cdata;
};

// Creates a new Python object wrapping the default at::Generator. The reference
// is borrowed. The caller should ensure that the at::Generator object lifetime
// last at least as long as the Python wrapper.
SMITH_PYTHON_API PyObject* THPGenerator_initDefaultGenerator(
    const at::Generator& cdata);

#define THPGenerasmitheck(obj) PyObject_IsInstance(obj, THPGeneratorClass)

SMITH_PYTHON_API extern PyObject* THPGeneratorClass;

bool THPGenerator_init(PyObject* module);

SMITH_PYTHON_API PyObject* THPGenerator_Wrap(const at::Generator& gen);

SMITH_PYTHON_API at::Generator THPGenerator_Unwrap(PyObject* state);

// Creates a new Python object for a Generator. The Generator must not already
// have a PyObject* associated with it.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
