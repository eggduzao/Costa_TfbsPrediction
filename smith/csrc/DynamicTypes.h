#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <smith/csrc/python_headers.h>

#include <ATen/Device.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <smith/csrc/Export.h>

#include <memory>
#include <string>

struct THPDtype;
struct THPLayout;

namespace c10 {
struct Storage;
}

namespace smith {
void registerDtypeObject(THPDtype* dtype, at::ScalarType scalarType);
void registerLayoutObject(THPLayout* thp_layout, at::Layout layout);

SMITH_PYTHON_API PyObject* createPyObject(const at::Storage& storage);
SMITH_PYTHON_API at::Storage createStorage(PyObject* obj);
SMITH_PYTHON_API std::tuple<at::Storage, at::ScalarType, bool>
createStorageGetType(PyObject* obj);
SMITH_PYTHON_API bool isStorage(PyObject* obj);

// Both methods below return a borrowed reference!
SMITH_PYTHON_API THPDtype* getTHPDtype(at::ScalarType scalarType);
SMITH_PYTHON_API THPLayout* getTHPLayout(at::Layout layout);
} // namespace smith
