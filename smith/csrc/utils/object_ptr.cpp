#include <c10/macros/Macros.h>
#include <smith/csrc/utils/object_ptr.h>

template <>
SMITH_PYTHON_API void THPPointer<PyObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;

template <>
SMITH_PYTHON_API void THPPointer<PyCodeObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyCodeObject>;

template <>
SMITH_PYTHON_API void THPPointer<PyFrameObject>::free() {
  if (ptr && C10_LIKELY(Py_IsInitialized()))
    Py_DECREF(ptr);
}

template class THPPointer<PyFrameObject>;
