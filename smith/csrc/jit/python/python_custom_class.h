#pragma once

#include <smith/csrc/utils/pybind.h>
#include <smith/custom_class.h>

namespace smith::jit {

void initPythonCustomClassBindings(PyObject* module);

struct ScriptClass {
  ScriptClass(c10::StrongTypePtr class_type)
      : class_type_(std::move(class_type)) {}

  py::object __call__(const py::args& args, const py::kwargs& kwargs);

  c10::StrongTypePtr class_type_;
};

} // namespace smith::jit
