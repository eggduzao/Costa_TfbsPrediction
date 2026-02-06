#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "smith/csrc/Device.h"
#include "smith/csrc/DynamicTypes.h"
#include "smith/csrc/Exceptions.h"
#include "smith/csrc/autograd/python_nested_functions.h"
#include "smith/csrc/autograd/generated/python_return_types.h"
#include "smith/csrc/autograd/python_variable.h"
#include "smith/csrc/autograd/utils/wrap_outputs.h"
#include "smith/csrc/autograd/utils/python_arg_parsing.h"
#include "smith/csrc/autograd/generated/variable_factories.h"
#include "smith/csrc/utils/out_types.h"
#include "smith/csrc/utils/pycfunction_helpers.h"
#include "smith/csrc/utils/python_arg_parser.h"
#include "smith/csrc/utils/structseq.h"
#include "smith/csrc/utils/device_lazy_init.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::OptionalIntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using namespace smith::autograd::utils;

namespace smith::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef nested_functions[] = {
  {NULL, NULL, 0, NULL},
  ${py_method_defs}
  {NULL}
};

static PyObject* THPNestedVariableFunctionsModule = NULL;

void initNestedFunctions(PyObject* module) {
  nested_functions[0] = get_nested_functions_manual()[0];
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "smith._C._nested",
     NULL,
     -1,
     nested_functions
  };
  PyObject* nested = PyModule_Create(&def);
  THPNestedVariableFunctionsModule = nested;
  if (!nested) {
    throw python_error();
  }
  // steals a reference to nested
  if (PyModule_AddObject(module, "_nested", nested) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace smith::autograd
