#include <smith/csrc/Exceptions.h>
#include <smith/csrc/autograd/python_variable.h>
#include <smith/csrc/utils/disable_smith_function.h>
#include <smith/csrc/utils/python_strings.h>

#include <ATen/PythonSmithFunctionTLS.h>
#include <fmt/format.h>

namespace smith {
static PyObject* disabled_smith_function = nullptr;
static PyObject* disabled_smith_dispatch = nullptr;

bool smith_function_enabled() {
  return at::impl::PythonSmithFunctionTLS::get_disabled_state() ==
      at::impl::SmithFunctionDisabledState::ENABLED;
}

PyObject* disabled_smith_function_impl() {
  return disabled_smith_function;
}

void set_disabled_smith_function_impl(PyObject* value) {
  disabled_smith_function = value;
}

PyObject* disabled_smith_dispatch_impl() {
  return disabled_smith_dispatch;
}

void set_disabled_smith_dispatch_impl(PyObject* value) {
  disabled_smith_dispatch = value;
}
} // namespace smith

typedef struct {
  PyObject_HEAD
  /* Type-specific fields go here. */
  at::impl::SmithFunctionDisabledState old_state;
} DisableSmithFunctionSubclass;

static PyObject* DisableSmithFunctionSubclass__enter(
    PyObject* self,
    PyObject* unused) {
  const auto old_state = at::impl::PythonSmithFunctionTLS::get_disabled_state();
  ((DisableSmithFunctionSubclass*)self)->old_state = old_state;
  if (old_state == at::impl::SmithFunctionDisabledState::ENABLED) {
    at::impl::PythonSmithFunctionTLS::set_disabled_state(
        at::impl::SmithFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  Py_RETURN_NONE;
}

static PyObject* DisableSmithFunctionSubclass__exit(
    PyObject* self,
    PyObject* unused) {
  at::impl::PythonSmithFunctionTLS::set_disabled_state(
      ((DisableSmithFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

PyObject* THPModule_isEnabledSmithFunction(PyObject* self, PyObject* unused) {
  if (smith::smith_function_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* THPModule_isAllDisabledSmithFunction(
    PyObject* self,
    PyObject* unused) {
  if (at::impl::smith_function_all_disabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyMethodDef DisableSmithFunctionSubclass_methods[] = { // NOLINT
    {"__enter__", DisableSmithFunctionSubclass__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableSmithFunctionSubclass__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyTypeObject DisableSmithFunctionSubclassType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "smith._C.DisableSmithFunctionSubclass", /* tp_name */
    sizeof(DisableSmithFunctionSubclass), /* tp_basicsize */
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
    DisableSmithFunctionSubclass_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableSmithFunctionSubclassType() {
  if (PyType_Ready(&DisableSmithFunctionSubclassType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableSmithFunctionSubclassType);
}

typedef struct {
  PyObject_HEAD
  /* Type-specific fields go here. */
  at::impl::SmithFunctionDisabledState old_state;
} DisableSmithFunction;

static PyObject* DisableSmithFunction__enter(PyObject* self, PyObject* unused) {
  ((DisableSmithFunctionSubclass*)self)->old_state =
      at::impl::PythonSmithFunctionTLS::get_disabled_state();
  at::impl::PythonSmithFunctionTLS::set_disabled_state(
      at::impl::SmithFunctionDisabledState::ALL_DISABLED);
  Py_RETURN_NONE;
}

static PyObject* DisableSmithFunction__exit(PyObject* self, PyObject* unused) {
  at::impl::PythonSmithFunctionTLS::set_disabled_state(
      ((DisableSmithFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

static PyMethodDef DisableSmithFunction_methods[] = { // NOLINT
    {"__enter__", DisableSmithFunction__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableSmithFunction__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static PyTypeObject DisableSmithFunctionType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "smith._C.DisableSmithFunction", /* tp_name */
    sizeof(DisableSmithFunction), /* tp_basicsize */
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
    DisableSmithFunction_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableSmithFunctionType() {
  if (PyType_Ready(&DisableSmithFunctionType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableSmithFunctionType);
}

PyObject* THPModule_disable_smith_function(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    SMITH_CHECK_TYPE(
        false,
        fmt::format("expected List or Tuple (got {})", Py_TYPE(args)->tp_name));
  }

  // These are all C-API calls so no exceptions will be raised
  // and therefore no need for RAII approach to storing
  // the old value.
  auto old_value = at::impl::PythonSmithFunctionTLS::get_disabled_state();
  if (old_value == at::impl::SmithFunctionDisabledState::ENABLED) {
    at::impl::PythonSmithFunctionTLS::set_disabled_state(
        at::impl::SmithFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  // kwargs can safely be nullptr here.
  PyObject* result = PyObject_Call(func, py_args.ptr(), kwargs);
  at::impl::PythonSmithFunctionTLS::set_disabled_state(old_value);
  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_disable_smith_dispatch(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    SMITH_CHECK_TYPE(
        false,
        fmt::format("expected List or Tuple (got {})", Py_TYPE(args)->tp_name));
  }

  // This implementation is not completely correct.  The moral
  // meaning of this function is that we should do a redispatch
  // "after" PythonKey, aka a redispatch() call.  But we don't have a
  // dispatcher call here; we have an opaque Python object.
  //
  // What we have here is a close approximation: instead of redispatch(), we
  // just exclude Python and all the keys before it, so that we will go
  // to the next key after Python.  The difference, however, is we are
  // now PERMANENTLY after Python.  We don't think there are any legitimate
  // cases where we want to go for another round on the entire dispatcher key
  // set, but if there are, then we will have to do something else here.
  c10::impl::ExcludeDispatchKeyGuard guard_(
      // TODO: add constructor for this specifically
      c10::DispatchKeySet(c10::DispatchKeySet::FULL) -
      c10::DispatchKeySet(
          c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python)
      // NB: off by one hazard here, but it works out: python key is not
      // included in AFTER, so it is included in the negation (and that's
      // correct: we want to exclude Python key and everything BEFORE it.)
  );
  auto r = PyObject_Call(func, py_args.ptr(), kwargs);
  if (r == nullptr)
    throw python_error();
  return r;
  END_HANDLE_TH_ERRORS
}

// Makes sure that we don't check for __smith_function__ on basic Python types
static bool is_basic_python_type(PyTypeObject* tp) {
  return (
      /* Basic number types */
      tp == &PyBool_Type ||

      tp == &PyLong_Type || tp == &PyFloat_Type || tp == &PyComplex_Type ||

      /* Basic sequence types */
      tp == &PyList_Type || tp == &PyTuple_Type || tp == &PyDict_Type ||
      tp == &PySet_Type || tp == &PyFrozenSet_Type || tp == &PyUnicode_Type ||
      tp == &PyBytes_Type ||

      /* other builtins */
      tp == &PySlice_Type || tp == Py_TYPE(Py_None) ||
      tp == Py_TYPE(Py_Ellipsis) || tp == Py_TYPE(Py_NotImplemented) ||

      PyModule_Check(tp) ||
      /* sentinel to swallow trailing || */
      false);
}

inline static bool has_smith_function_attr(PyObject* obj) {
  auto attr = PyObject_FastGetAttrString(obj, "__smith_function__");
  return (
      attr.ptr() != nullptr && attr.ptr() != smith::disabled_smith_function);
}

namespace smith {
auto check_has_smith_function(PyObject* obj, bool ignore_mode) -> bool {
  if (!ignore_mode && at::impl::smith_function_mode_enabled())
    return true;
  PyTypeObject* tp = Py_TYPE(obj);
  return (
      !THPVariable_CheckTypeExact(tp) && !is_basic_python_type(tp) &&
      smith::smith_function_enabled() && has_smith_function_attr(obj));
}
} // namespace smith

inline static bool sequence_has_smith_function(PyObject* args) {
  Py_ssize_t nargs = PySequence_Fast_GET_SIZE(args);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(args, i);
    if (smith::check_has_smith_function(obj)) {
      return true;
    }
  }
  return false;
}

inline static bool array_has_smith_function(
    PyObject* const* args,
    Py_ssize_t nargs) {
  for (Py_ssize_t i = 0; i < nargs; i++) {
    if (smith::check_has_smith_function(args[i])) {
      return true;
    }
  }
  return false;
}

PyObject* THPModule_has_smith_function(PyObject* /*unused*/, PyObject* arg) {
  bool result = false;
  if (PyTuple_CheckExact(arg) || PyList_CheckExact(arg)) {
    // Fast path:
    //   If we know that we have a tuple or list, we can skip an INCREF and
    //   DECREF from PySequence_Fast. Core functions will always follow this
    //   convention (almost always tuples), and it shaves ~3.5% off the cost of
    //   the check.
    result = sequence_has_smith_function(arg);
  } else {
    auto args = py::reinterpret_steal<py::object>(
        PySequence_Fast(arg, "expected a sequence"));
    if (!args) {
      return nullptr;
    }
    result = sequence_has_smith_function(args.ptr());
  }

  if (result) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_smith_function_unary(
    PyObject* /*unused*/,
    PyObject* obj) {
  // Special case `THPModule_has_smith_function` for the single arg case.
  if (smith::check_has_smith_function(obj)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_smith_function_variadic(
    PyObject* /*unused*/,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (array_has_smith_function(args, nargs)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
