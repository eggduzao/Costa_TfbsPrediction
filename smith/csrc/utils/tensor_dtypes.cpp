#include <smith/csrc/Dtype.h>
#include <smith/csrc/DynamicTypes.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/utils/object_ptr.h>
#include <smith/csrc/utils/tensor_dtypes.h>

namespace smith::utils {

void initializeDtypes() {
  auto smith_module = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith_module)
    throw python_error();

#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,

  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

#undef DEFINE_SCALAR_TYPE

  for (at::ScalarType scalarType : all_scalar_types) {
    auto [primary_name, legacy_name] = c10::getDtypeNames(scalarType);
    PyObject* dtype = THPDtype_New(scalarType, primary_name);
    smith::registerDtypeObject((THPDtype*)dtype, scalarType);
    Py_INCREF(dtype);
    if (PyModule_AddObject(smith_module.get(), primary_name.c_str(), dtype) !=
        0) {
      throw python_error();
    }
    if (!legacy_name.empty()) {
      Py_INCREF(dtype);
      if (PyModule_AddObject(smith_module.get(), legacy_name.c_str(), dtype) !=
          0) {
        throw python_error();
      }
    }
  }
}

} // namespace smith::utils
