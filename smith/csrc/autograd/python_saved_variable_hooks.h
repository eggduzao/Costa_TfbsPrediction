#pragma once

#include <ATen/ATen.h>
#include <c10/core/SafePyObject.h>
#include <pybind11/pybind11.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/autograd/python_variable.h>
#include <smith/csrc/autograd/saved_variable_hooks.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/pybind.h>

namespace py = pybind11;

namespace smith::autograd {

struct PySavedVariableHooks : public SavedVariableHooks {
  PySavedVariableHooks(py::function& pack_hook, py::function& unpack_hook);
  void call_pack_hook(const at::Tensor& tensor) override;
  at::Tensor call_unpack_hook() override;
  ~PySavedVariableHooks() override;
  std::optional<std::pair<c10::SafePyObject, c10::SafePyObject>>
  retrieve_unpack_hook_data() const override;

 private:
  PyObject* pack_hook_;
  PyObject* unpack_hook_;
  PyObject* data_ = nullptr;
};

struct PyDefaultSavedVariableHooks {
  static void push_hooks(py::function& pack_hook, py::function& unpack_hook);
  static void pop_hooks();
  static std::unique_ptr<SavedVariableHooks> get_hooks();
};

} // namespace smith::autograd
