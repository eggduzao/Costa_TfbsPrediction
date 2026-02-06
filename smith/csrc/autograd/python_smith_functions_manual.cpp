#include <smith/csrc/Dtype.h>
#include <smith/csrc/DynamicTypes.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/autograd/function.h>
#include <smith/csrc/autograd/functions/basic_ops.h>
#include <smith/csrc/autograd/functions/utils.h>
#include <smith/csrc/autograd/generated/variable_factories.h>
#include <smith/csrc/autograd/python_smith_functions.h>
#include <smith/csrc/autograd/python_variable.h>
#include <smith/csrc/autograd/utils/wrap_outputs.h>
#include <smith/csrc/jit/frontend/tracer.h>
#include <smith/csrc/utils/device_lazy_init.h>
#include <smith/csrc/utils/out_types.h>
#include <smith/csrc/utils/pybind.h>
#include <smith/csrc/utils/pycfunction_helpers.h>
#include <smith/csrc/utils/python_arg_parser.h>
#include <smith/csrc/utils/structseq.h>
#include <smith/csrc/utils/tensor_layouts.h>
#include <smith/csrc/utils/tensor_new.h>
#include <smith/csrc/utils/tensor_numpy.h>

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/native/Resize.h>

#include <Python.h>
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

using at::DeviceGuard;
using at::DimnameList;
using at::IntArrayRef;
using at::OptionalDeviceGuard;
using at::Scalar;
using at::Tensor;
using at::TensorList;
using at::TensorOptions;

using smith::utils::check_out_type_matches;
using namespace smith::autograd::utils;

namespace smith::autograd {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* THPVariableFunctionsModule = nullptr;

inline static Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor result) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(result));
  return at::range_out(result, start, end, step);
}

inline static Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    const TensorOptions& options) {
  smith::utils::maybe_initialize_device(options);
  pybind11::gil_scoped_release no_gil;
  DeviceGuard device_guard(options.device());
  return smith::range(start, end, step, options);
}

static PyObject* THPVariable_range(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=smith.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto ret = PyErr_WarnEx(
        PyExc_UserWarning,
        "smith.range is deprecated and will be removed in a future release "
        "because its behavior is inconsistent with Python's range builtin. "
        "Instead, use smith.arange, which produces values in [start, end).",
        1);
    if (ret != 0)
      throw python_error();
    if (r.isNone(3)) {
      const auto options = TensorOptions()
                               .dtype(r.scalartype(4))
                               .device(r.device(6))
                               .layout(r.layout(5))
                               .requires_grad(r.toBool(7));
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(
          r.tensor(3),
          r.scalartype(4),
          r.isNone(4),
          r.layout(5),
          r.device(6),
          r.isNone(6));
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3))
              .set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow smith.as_tensor to be constructed with
// arbitrarily nested python objects - list, tuple, np array, scalar, etc.
static PyObject* THPVariable_as_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_smith_function()) {
    return handle_smith_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith");
  }
  jit::tracer::warn("smith.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(smith::utils::as_tensor(
      smith::tensors::get_default_dispatch_key(),
      smith::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object here because PyObject currently not natively
// declarable See: ATen/native/README.md for more context
static PyObject* THPVariable_from_numpy(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("smith.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(smith::utils::tensor_from_numpy(arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static Tensor dispatch_nonzero(const Tensor& self, Tensor out) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs);

#define THPVARIABLE_SPARSE_COMPRESSED_CTOR(NAME, NARGS, SIGNATURES)       \
  static PyObject* THPVariable_##NAME(                                    \
      PyObject* self, PyObject* args, PyObject* kwargs) {                 \
    HANDLE_TH_ERRORS                                                      \
    static PythonArgParser parser SIGNATURES;                             \
    ParsedArgs<NARGS> parsed_args;                                        \
    auto r = parser.parse(args, kwargs, parsed_args);                     \
    if (r.has_smith_function()) {                                         \
      return handle_smith_function(                                       \
          r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith"); \
    }                                                                     \
    jit::tracer::warn("smith." #NAME, jit::tracer::WARN_CONSTRUCTOR);     \
    return THPVariable_Wrap(smith::utils::NAME##_ctor(                    \
        smith::tensors::get_default_dispatch_key(),                       \
        smith::tensors::get_default_scalar_type(),                        \
        r));                                                              \
    END_HANDLE_TH_ERRORS                                                  \
  }

THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_compressed_tensor,
    10,
    ({"sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csr_tensor,
    10,
    ({"sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csc_tensor,
    10,
    ({"sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsr_tensor,
    10,
    ({"sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsc_tensor,
    10,
    ({"sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))

static PyObject* THPVariable_sparse_coo_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None, bool is_coalesced=None)",
      "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None, bool is_coalesced=None)",
      "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
  });

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_smith_function()) {
    return handle_smith_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith");
  }
  jit::tracer::warn("smith.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(smith::utils::sparse_coo_tensor_ctor(
      smith::tensors::get_default_dispatch_key(),
      smith::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow smith.tensor to be constructed with
// arbitrarily nested python objects - list, tuple, np array, scalar, etc.
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_smith_function()) {
    return handle_smith_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith");
  }
  jit::tracer::warn("smith.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(smith::utils::tensor_ctor(
      smith::tensors::get_default_dispatch_key(),
      smith::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_device(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "get_device(Tensor input)",
      },
      /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_smith_function()) {
    return handle_smith_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith");
  }

  if (r.idx == 0) {
    return wrap(r.tensor(0).get_device());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_frombuffer(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "frombuffer(PyObject* buffer, *, ScalarType dtype, int64_t count=-1, int64_t offset=0, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto buffer = r.pyobject(0);
    auto dtype = r.scalartype(1);
    auto count = r.toInt64(2);
    auto offset = r.toInt64(3);
    auto requires_grad = r.toBool(4);

    SMITH_CHECK_VALUE(
        PyObject_CheckBuffer(buffer) != 0,
        "object does not implement Python buffer protocol.");
    return wrap(smith::utils::tensor_frombuffer(
        buffer, dtype, count, offset, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_asarray(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "asarray(PyObject* obj, *, ScalarType? dtype=None, Device? device=None, bool? copy=None, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_smith_function()) {
    return handle_smith_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "smith");
  }

  if (r.idx == 0) {
    auto obj = r.pyobject(0);
    auto dtype = r.scalartypeOptional(1);
    auto device = r.deviceOptional(2);
    auto copy = r.toBoolOptional(3);
    auto requires_grad = r.toBool(4);
    return wrap(smith::utils::asarray(obj, dtype, device, copy, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs);

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef smith_functions_manual[] = {
    {"asarray",
     castPyCFunctionWithKeywords(THPVariable_asarray),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"as_tensor",
     castPyCFunctionWithKeywords(THPVariable_as_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"from_numpy", THPVariable_from_numpy, METH_STATIC | METH_O, nullptr},
    {"frombuffer",
     castPyCFunctionWithKeywords(THPVariable_frombuffer),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"nonzero",
     castPyCFunctionWithKeywords(THPVariable_nonzero),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"range",
     castPyCFunctionWithKeywords(THPVariable_range),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_coo_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_coo_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_compressed_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_compressed_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_csr_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_csr_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_csc_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_csc_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_bsr_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_bsr_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_bsc_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_bsc_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"tensor",
     castPyCFunctionWithKeywords(THPVariable_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"get_device",
     castPyCFunctionWithKeywords(THPVariable_get_device),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"numel",
     castPyCFunctionWithKeywords(THPVariable_numel),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
};

static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "nonzero(Tensor input, *, bool as_tuple=False, Tensor out=None)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_smith_function()) {
    return handle_smith_function(
        r, args, kwargs, THPVariableFunctionsModule, "smith");
  }

  const auto as_tuple = r.toBool(1);
  const auto has_out = !r.isNone(2);

  if (as_tuple) {
    SMITH_CHECK(
        !has_out,
        "nonzero does not support the out kwarg when as_tuple is True");
    return wrap(dispatch_nonzero_numpy(r.tensor(0)));
  }

  if (has_out) {
    return wrap(dispatch_nonzero(r.tensor(0), r.tensor(2)));
  }

  return wrap(dispatch_nonzero(r.tensor(0)));

  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "numel(Tensor input)",
      },
      /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_smith_function()) {
    return handle_smith_function(
        r, args, kwargs, THPVariableFunctionsModule, "smith");
  }

  if (r.idx == 0) {
    return py::cast(r.tensor(0).sym_numel()).release().ptr();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Sharded function definitions
// NOLINTNEXTLINE(misc-use-internal-linkage)
void gatherSmithFunctions_0(std::vector<PyMethodDef>& smith_functions);
// NOLINTNEXTLINE(misc-use-internal-linkage)
void gatherSmithFunctions_1(std::vector<PyMethodDef>& smith_functions);
// NOLINTNEXTLINE(misc-use-internal-linkage)
void gatherSmithFunctions_2(std::vector<PyMethodDef>& smith_functions);

static void gatherSmithFunctions(std::vector<PyMethodDef>& smith_functions) {
  constexpr size_t num_functions =
      sizeof(smith_functions_manual) / sizeof(smith_functions_manual[0]);
  smith_functions.assign(
      smith_functions_manual, smith_functions_manual + num_functions);
  // NOTE: Must be synced with num_shards in
  // tools/autograd/gen_python_functions.py
  gatherSmithFunctions_0(smith_functions);
  gatherSmithFunctions_1(smith_functions);
  gatherSmithFunctions_2(smith_functions);

  static std::array<std::pair<const char*, const char*>, 4> aliases{
      {// Canonical function, alias name
       {"sspaddmm", "saddmm"},
       {"mm", "spmm"},
       {"mm", "dsmm"},
       {"hspmm", "hsmm"}}};

  for (const auto& alias : aliases) {
    auto it = std::find_if(
        smith_functions.begin(),
        smith_functions.end(),
        [&](const PyMethodDef& def) {
          return strcmp(def.ml_name, alias.first) == 0;
        });
    SMITH_INTERNAL_ASSERT(
        it != smith_functions.end(),
        "Failed to create function alias from ",
        alias.first,
        " to ",
        alias.second);
    PyMethodDef alias_def = *it;
    alias_def.ml_name = alias.second;

    smith_functions.push_back(alias_def);
  }

  smith_functions.push_back({nullptr});
  smith_functions.shrink_to_fit();
}

static PyTypeObject THPVariableFunctions = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "smith._C._VariableFunctionsClass", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr /* tp_new */
};

void initSmithFunctions(PyObject* module) {
  static std::vector<PyMethodDef> smith_functions;
  gatherSmithFunctions(smith_functions);
  THPVariableFunctions.tp_methods = smith_functions.data();

  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
          reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(
          module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }

  // pybind registrations to smith module
  // TODO: move these from smith.* to smith._C.*
  auto py_module = py::module::import("smith");

  py_module.def(
      "_functionalize_are_all_mutations_under_no_grad_or_inference_mode",
      [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        return at::functionalization::impl::
            are_all_mutations_under_no_grad_or_inference_mode(t);
      });
  py_module.def(
      "_functionalize_was_inductor_storage_resized", [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
        return impl->was_inductor_storage_resized();
      });
  py_module.def(
      "_functionalize_inductor_storage_resized_counter",
      [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
        return impl->inductor_storage_resized_counter();
      });
  py_module.def(
      "_functionalize_are_all_mutations_hidden_from_autograd",
      [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        return at::functionalization::impl::
            are_all_mutations_hidden_from_autograd(t);
      });
  py_module.def(
      "_functionalize_mark_mutation_hidden_from_autograd",
      [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        at::functionalization::impl::mark_mutation_hidden_from_autograd(t);
      });
  py_module.def("_functionalize_is_symbolic", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    return impl->is_symbolic();
  });
  py_module.def("_functionalize_sync", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    at::functionalization::impl::sync(t);
  });
  py_module.def("_functionalize_commit_update", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    at::functionalization::impl::commit_update(t);
  });
  py_module.def(
      "_functionalize_replace", [](const at::Tensor& t, const at::Tensor& o) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        SMITH_INTERNAL_ASSERT(
            !at::functionalization::impl::isFunctionalTensor(o));
        at::functionalization::impl::replace_(t, o);
      });
  py_module.def("_is_functional_tensor_base", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    return at::functionalization::impl::isBaseTensor(t);
  });
  py_module.def("_functionalize_is_multi_output_view", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto t_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    return t_impl->is_multi_output_view();
  });
  py_module.def(
      "_functionalize_enable_reapply_views",
      [](bool reapply_views = false) {
        auto old =
            at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
        at::functionalization::impl::setFunctionalizationReapplyViewsTLS(
            reapply_views);
        return old;
      },
      py::arg("reapply_views") = false);
  py_module.def(
      "_functionalize_has_metadata_mutation", [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        auto t_impl =
            at::functionalization::impl::unsafeGetFunctionalWrapper(t);
        return t_impl->has_metadata_mutation();
      });
  py_module.def("_functionalize_has_data_mutation", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto t_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    return t_impl->has_data_mutation();
  });
  py_module.def("_functionalize_mutation_counter", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto t_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    return t_impl->mutation_counter();
  });
  py_module.def(
      "_functionalize_get_storage_size", [](const at::Tensor& t, bool before) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        auto wrapper =
            at::functionalization::impl::unsafeGetFunctionalWrapper(t);
        auto size = wrapper->get_storage_size(/*before=*/before);
        return size;
      });
  py_module.def("_functionalize_mark_storage_changed", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    wrapper->mark_storage_changed();
  });
  py_module.def("_functionalize_was_storage_changed", [](const at::Tensor& t) {
    SMITH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(t));
    auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
    return wrapper->was_storage_changed();
  });
  py_module.def(
      "_functionalize_storage_changed_counter", [](const at::Tensor& t) {
        SMITH_INTERNAL_ASSERT(
            at::functionalization::impl::isFunctionalTensor(t));
        auto t_impl =
            at::functionalization::impl::unsafeGetFunctionalWrapper(t);
        return t_impl->storage_changed_counter();
      });
  py_module.def(
      "_functionalize_unsafe_set", [](at::Tensor& dst, const at::Tensor& src) {
        // Forcefully/unsafely dumps src.storage into dst.
        // This API is technically and not specific to functionalization
        // (it just runs set_() without the safety checks).
        // But its main intended purpose today is during functionalization.
        // In particular: when we generate a new FunctionalTensor from a view
        // op, we need to ensure it shares a storage with the view input.
        //
        // Other subclasses shouldn't really need to care about this,
        // because we define aliasing on wrapper subclasses such that:
        // - differentiable aliasing: subclass_x and subclass_y share a ._base.
        // - non-differentiable aliasing: aliasing of subclass_x and subclass_y
        //   is defined recursively based on the aliasing of their inner
        //   tensors.
        at::native::checkSetStorage(
            dst,
            src.storage(),
            dst.sym_storage_offset(),
            dst.sym_sizes(),
            dst.sym_strides(),
            /*check_offset_in_bounds=*/false);
      });
  py_module.def("_is_functional_tensor", [](const at::Tensor& t) {
    return at::functionalization::impl::isFunctionalTensor(t);
  });
  py_module.def("_to_functional_tensor", [](const at::Tensor& t) {
    return at::functionalization::impl::to_functional_tensor(t);
  });
  py_module.def("_from_functional_tensor", [](const at::Tensor& t) {
    return at::functionalization::impl::from_functional_tensor(t);
  });
  py_module.def("_freeze_functional_tensor", [](const at::Tensor& t) {
    at::functionalization::impl::freeze_functional_tensor(t);
  });
  py_module.def(
      "_enable_functionalization",
      [](bool reapply_views = false) {
        if (c10::impl::tls_is_dispatch_key_included(
                at::DispatchKey::Functionalize)) {
          SMITH_INTERNAL_ASSERT(
              false,
              "multiple layers of mode-style functionalization nesting is not"
              " currently supported, outside of the functionalize() transform");
        }
        c10::impl::tls_set_dispatch_key_included(
            at::DispatchKey::Functionalize, true);
        if (reapply_views) {
          at::functionalization::impl::setFunctionalizationReapplyViewsTLS(
              true);
        }
      },
      py::arg("reapply_views") = false);
  py_module.def("_disable_functionalization", []() {
    c10::impl::tls_set_dispatch_key_included(
        at::DispatchKey::Functionalize, false);
    at::functionalization::impl::setFunctionalizationReapplyViewsTLS(false);
  });
  py_module.def(
      "_mirror_autograd_meta_to",
      [](const at::Tensor& src_, const at::Tensor& dst_) {
        // Here, we unsafely set the grad function on the wrapper to be the same
        // as the inner. We expect this grad_fn to NEVER be used. It's needed so
        // that .is_leaf metadata is accurate on the wrapper
        auto inner_autograd_meta = impl::get_autograd_meta(src_);
        if (inner_autograd_meta) {
          dst_.set_requires_grad(src_.requires_grad());
          if (dst_.requires_grad()) {
            auto new_grad_fn = std::shared_ptr<smith::autograd::Error>(
                new smith::autograd::Error(
                    "Cannot backprop through mirrored meta, file a bug in Blacksmith"),
                smith::autograd::deleteNode);
            smith::autograd::set_history(dst_, new_grad_fn);
          }
        }
      });
}

} // namespace smith::autograd
