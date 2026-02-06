#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/python_headers.h>

#include <ATen/Device.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct SMITH_API THPDevice {
  PyObject_HEAD
  at::Device device;
};

SMITH_API extern PyTypeObject THPDeviceType;

inline bool THPDevice_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

SMITH_API PyObject* THPDevice_New(const at::Device& device);

SMITH_API void THPDevice_init(PyObject* module);
