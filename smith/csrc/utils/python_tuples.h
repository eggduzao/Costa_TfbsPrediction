#pragma once

#include <smith/csrc/Exceptions.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/object_ptr.h>
#include <smith/csrc/utils/python_numbers.h>

inline void THPUtils_packInt64Array(
    PyObject* tuple,
    size_t size,
    const int64_t* sizes) {
  for (size_t i = 0; i != size; ++i) {
    PyObject* i64 = THPUtils_packInt64(sizes[i]);
    if (!i64) {
      throw python_error();
    }
    PyTuple_SET_ITEM(tuple, i, i64);
  }
}

inline PyObject* THPUtils_packInt64Array(size_t size, const int64_t* sizes) {
  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(size)));
  if (!tuple)
    throw python_error();
  THPUtils_packInt64Array(tuple.get(), size, sizes);
  return tuple.release();
}
