#pragma once

#include <smith/detail/static.h>
#include <smith/nn/module.h>
#include <smith/ordered_dict.h>
#include <smith/types.h>

#include <smith/csrc/Device.h>
#include <smith/csrc/Dtype.h>
#include <smith/csrc/DynamicTypes.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/autograd/python_variable.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/pybind.h>
#include <smith/csrc/utils/python_numbers.h>
#include <smith/csrc/utils/python_tuples.h>

#include <iterator>
#include <string>
#include <utility>

namespace smith::python {
namespace detail {
inline Device py_object_to_device(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDevice_Check(obj)) {
    return reinterpret_cast<THPDevice*>(obj)->device;
  }
  SMITH_CHECK_TYPE(false, "Expected device");
}

inline Dtype py_object_to_dtype(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDtype_Check(obj)) {
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }
  SMITH_CHECK_TYPE(false, "Expected dtype");
}

template <typename ModuleType>
using PyModuleClass =
    py::class_<ModuleType, smith::nn::Module, std::shared_ptr<ModuleType>>;

/// Dynamically creates a subclass of `smith.nn.cpp.ModuleWrapper` that is also
/// a subclass of `smith.nn.Module`, and passes it the user-provided C++ module
/// to which it delegates all calls.
template <typename ModuleType>
void bind_cpp_module_wrapper(
    const py::module& module,
    PyModuleClass<ModuleType> cpp_class,
    const char* name) {
  // Grab the `smith.nn.cpp.ModuleWrapper` class, which we'll subclass
  // with a dynamically created class below.
  py::object cpp_module =
      py::module::import("smith.nn.cpp").attr("ModuleWrapper");

  // Grab the `type` class which we'll use as a metaclass to create a new class
  // dynamically.
  py::object type_metaclass =
      py::reinterpret_borrow<py::object>((PyObject*)&PyType_Type);

  // The `ModuleWrapper` constructor copies all functions to its own `__dict__`
  // in its constructor, but we do need to give our dynamic class a constructor.
  // Inside, we construct an instance of the original C++ module we're binding
  // (the `smith::nn::Module` subclass), and then forward it to the
  // `ModuleWrapper` constructor.
  py::dict attributes;

  // `type()` always needs a `str`, but pybind11's `str()` method always creates
  // a `unicode` object.
  py::object name_str = py::str(name);

  // Dynamically create the subclass of `ModuleWrapper`, which is a subclass of
  // `smith.nn.Module`, and will delegate all calls to the C++ module we're
  // binding.
  py::object wrapper_class =
      type_metaclass(name_str, py::make_tuple(cpp_module), attributes);

  // The constructor of the dynamic class calls `ModuleWrapper.__init__()`,
  // which replaces its methods with those of the C++ module.
  wrapper_class.attr("__init__") = py::cpp_function(
      [cpp_module, cpp_class](
          const py::object& self,
          const py::args& args,
          const py::kwargs& kwargs) {
        cpp_module.attr("__init__")(self, cpp_class(*args, **kwargs));
      },
      py::is_method(wrapper_class));

  // Calling `my_module.my_class` now means that `my_class` is a subclass of
  // `ModuleWrapper`, and whose methods call into the C++ module we're binding.
  module.attr(name) = wrapper_class;
}
} // namespace detail

/// Adds method bindings for a pybind11 `class_` that binds an `nn::Module`
/// subclass.
///
/// Say you have a pybind11 class object created with `py::class_<Net>(m,
/// "Net")`. This function will add all the necessary `.def()` calls to bind the
/// `nn::Module` base class' methods, such as `train()`, `eval()` etc. into
/// Python.
///
/// Users should prefer to use `bind_module` if possible.
template <typename ModuleType, typename... Extra>
py::class_<ModuleType, Extra...> add_module_bindings(
    py::class_<ModuleType, Extra...> module) {
  // clang-format off
  return module
      .def("train",
          [](ModuleType& module, bool mode) { module.train(mode); },
          py::arg("mode") = true)
      .def("eval", [](ModuleType& module) { module.eval(); })
      .def("clone", [](ModuleType& module) { return module.clone(); })
      .def_property_readonly(
          "training", [](ModuleType& module) { return module.is_training(); })
      .def("zero_grad", [](ModuleType& module) { module.zero_grad(); })
      .def_property_readonly( "_parameters", [](ModuleType& module) {
            return module.named_parameters(/*recurse=*/false);
          })
      .def("parameters", [](ModuleType& module, bool recurse) {
            return module.parameters(recurse);
          },
          py::arg("recurse") = true)
      .def("named_parameters", [](ModuleType& module, bool recurse) {
            return module.named_parameters(recurse);
          },
          py::arg("recurse") = true)
      .def_property_readonly("_buffers", [](ModuleType& module) {
            return module.named_buffers(/*recurse=*/false);
          })
      .def("buffers", [](ModuleType& module, bool recurse) {
            return module.buffers(recurse); },
          py::arg("recurse") = true)
      .def("named_buffers", [](ModuleType& module, bool recurse) {
            return module.named_buffers(recurse);
          },
          py::arg("recurse") = true)
      .def_property_readonly(
        "_modules", [](ModuleType& module) { return module.named_children(); })
      .def("modules", [](ModuleType& module) { return module.modules(); })
      .def("named_modules",
           [](ModuleType& module, const py::object& /* unused */, std::string prefix, bool remove_duplicate /* unused */) {
            return module.named_modules(std::move(prefix));
          },
          py::arg("memo") = py::none(),
          py::arg("prefix") = std::string(),
          py::arg("remove_duplicate") = true)
      .def("children", [](ModuleType& module) { return module.children(); })
      .def("named_children",
          [](ModuleType& module) { return module.named_children(); })
      .def("to", [](ModuleType& module, py::object object, bool non_blocking) {
            if (THPDevice_Check(object.ptr())) {
              module.to(
                  reinterpret_cast<THPDevice*>(object.ptr())->device,
                  non_blocking);
            } else {
              module.to(detail::py_object_to_dtype(object), non_blocking);
            }
          },
          py::arg("dtype_or_device"),
          py::arg("non_blocking") = false)
      .def("to",
          [](ModuleType& module,
             const py::object& device,
             const py::object& dtype,
             bool non_blocking) {
              if (device.is_none()) {
                module.to(detail::py_object_to_dtype(dtype), non_blocking);
              } else if (dtype.is_none()) {
                module.to(detail::py_object_to_device(device), non_blocking);
              } else {
                module.to(
                    detail::py_object_to_device(device),
                    detail::py_object_to_dtype(dtype),
                    non_blocking);
              }
          },
          py::arg("device"),
          py::arg("dtype"),
          py::arg("non_blocking") = false)
      .def("cuda", [](ModuleType& module) { module.to(kCUDA); })
      .def("cpu", [](ModuleType& module) { module.to(kCPU); })
      .def("float", [](ModuleType& module) { module.to(kFloat32); })
      .def("double", [](ModuleType& module) { module.to(kFloat64); })
      .def("half", [](ModuleType& module) { module.to(kFloat16); })
      .def("__str__", [](ModuleType& module) { return module.name(); })
      .def("__repr__", [](ModuleType& module) { return module.name(); });
  // clang-format on
}

/// Creates a pybind11 class object for an `nn::Module` subclass type and adds
/// default bindings.
///
/// After adding the default bindings, the class object is returned, such that
/// you can add more bindings.
///
/// Example usage:
/// \rst
/// .. code-block:: cpp
///
///   struct Net : smith::nn::Module {
///     Net(int in, int out) { }
///     smith::Tensor forward(smith::Tensor x) { return x; }
///   };
///
///   PYBIND11_MODULE(my_module, m) {
///     smith::python::bind_module<Net>(m, "Net")
///       .def(py::init<int, int>())
///       .def("forward", &Net::forward);
///  }
/// \endrst
template <typename ModuleType, bool force_enable = false>
std::enable_if_t<
    !smith::detail::has_forward<ModuleType>::value || force_enable,
    detail::PyModuleClass<ModuleType>>
bind_module(py::module module, const char* name) {
  py::module cpp = module.def_submodule("cpp");
  auto cpp_class =
      add_module_bindings(detail::PyModuleClass<ModuleType>(cpp, name));
  detail::bind_cpp_module_wrapper(module, cpp_class, name);
  return cpp_class;
}

/// Creates a pybind11 class object for an `nn::Module` subclass type and adds
/// default bindings.
///
/// After adding the default bindings, the class object is returned, such that
/// you can add more bindings.
///
/// If the class has a `forward()` method, it is automatically exposed as
/// `forward()` and `__call__` in Python.
///
/// Example usage:
/// \rst
/// .. code-block:: cpp
///
///   struct Net : smith::nn::Module {
///     Net(int in, int out) { }
///     smith::Tensor forward(smith::Tensor x) { return x; }
///   };
///
///   PYBIND11_MODULE(my_module, m) {
///     smith::python::bind_module<Net>(m, "Net")
///       .def(py::init<int, int>())
///       .def("forward", &Net::forward);
///  }
/// \endrst
template <
    typename ModuleType,
    typename = std::enable_if_t<smith::detail::has_forward<ModuleType>::value>>
detail::PyModuleClass<ModuleType> bind_module(
    py::module module,
    const char* name) {
  return bind_module<ModuleType, /*force_enable=*/true>(module, name)
      .def("forward", &ModuleType::forward)
      .def("__call__", &ModuleType::forward);
}
} // namespace smith::python
