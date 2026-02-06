#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <smith/csrc/python_headers.h>

namespace smith {
// Sometimes we don't want infinite recursion for subclasses,
// Or a way to achieve the old behaviour.

// This is an internal utility, not exposed to users.
bool smith_function_enabled();
PyObject* disabled_smith_function_impl();
PyObject* disabled_smith_dispatch_impl();
void set_disabled_smith_function_impl(PyObject* value);
void set_disabled_smith_dispatch_impl(PyObject* value);
// Set ignore_mode to true if you're trying to collect overloaded arguments;
// using mode here will improperly cause you to add ALL objects to the
// overloaded list even if they don't actually have __smith_function__
bool check_has_smith_function(PyObject* obj, bool ignore_mode = false);

struct DisableSmithDispatch {
  DisableSmithDispatch()
      : guard_(c10::DispatchKeySet(
            {c10::DispatchKey::Python, c10::DispatchKey::PreDispatch})),
        guard_tls_snapshot_(c10::DispatchKey::PythonTLSSnapshot) {}
  c10::impl::ExcludeDispatchKeyGuard guard_;
  c10::impl::ExcludeDispatchKeyGuard guard_tls_snapshot_;
};

} // namespace smith

PyObject* THPModule_isEnabledSmithFunction(PyObject* self, PyObject* unused);
PyObject* THPModule_isAllDisabledSmithFunction(
    PyObject* self,
    PyObject* unused);
PyObject* THPModule_DisableSmithFunctionType();
PyObject* THPModule_DisableSmithFunctionSubclassType();
PyObject* THPModule_disable_smith_function(PyObject* self, PyObject* args);
PyObject* THPModule_disable_smith_dispatch(PyObject* self, PyObject* args);
PyObject* THPModule_has_smith_function(PyObject* /*unused*/, PyObject* arg);
PyObject* THPModule_has_smith_function_unary(
    PyObject* /*unused*/,
    PyObject* obj);
PyObject* THPModule_has_smith_function_variadic(
    PyObject* /*unused*/,
    PyObject* const* args,
    Py_ssize_t nargs);
