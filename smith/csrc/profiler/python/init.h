#pragma once

#include <Python.h>

#include <smith/csrc/profiler/collection.h>
#include <smith/csrc/profiler/python/pybind.h>

namespace pybind11::detail {
using smith::profiler::impl::TensorID;

#define STRONG_POINTER_TYPE_CASTER(T) \
  template <>                         \
  struct type_caster<T> : public strong_pointer_type_caster<T> {};

STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::StorageImplData)
STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::AllocationID)
STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::TensorImplAddress)
STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::PyModuleSelf)
STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::PyModuleCls)
STRONG_POINTER_TYPE_CASTER(smith::profiler::impl::PyOptimizerSelf)
#undef STRONG_POINTER_TYPE_CASTER

template <>
struct type_caster<TensorID> : public strong_uint_type_caster<TensorID> {};
} // namespace pybind11::detail

namespace smith::profiler {

void initPythonBindings(PyObject* module);

} // namespace smith::profiler
