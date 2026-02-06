#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "smith/csrc/Device.h"
#include "smith/csrc/DynamicTypes.h"
#include "smith/csrc/Exceptions.h"
#include "smith/csrc/autograd/python_fft_functions.h"
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

#include <ATen/core/Tensor.h>

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
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using smith::utils::check_out_type_matches;
using namespace smith::autograd::utils;

namespace smith::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef fft_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPFFTVariableFunctionsModule = NULL;

void initFFTFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "smith._C._fft",
     NULL,
     -1,
     fft_functions
  };
  PyObject* fft = PyModule_Create(&def);
  THPFFTVariableFunctionsModule = fft;
  if (!fft) {
    throw python_error();
  }
  // steals a reference to fft
  if (PyModule_AddObject(module, "_fft", fft) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace smith::autograd
