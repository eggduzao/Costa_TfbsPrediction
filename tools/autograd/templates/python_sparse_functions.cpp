#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "smith/csrc/Device.h"
#include "smith/csrc/DynamicTypes.h"
#include "smith/csrc/Exceptions.h"
#include "smith/csrc/autograd/python_sparse_functions.h"
#include "smith/csrc/autograd/python_variable.h"
#include "smith/csrc/autograd/utils/wrap_outputs.h"
#include "smith/csrc/autograd/utils/python_arg_parsing.h"
#include "smith/csrc/utils/pycfunction_helpers.h"
#include "smith/csrc/utils/python_arg_parser.h"
#include "smith/csrc/utils/structseq.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

using namespace smith::autograd::utils;

namespace smith::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef sparse_functions[] = {
  ${py_method_defs}
  {NULL}
};

static PyObject* THPSparseVariableFunctionsModule = NULL;

void initSparseFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "smith._C._sparse",
     NULL,
     -1,
     sparse_functions
  };
  PyObject* sparse = PyModule_Create(&def);
  THPSparseVariableFunctionsModule = sparse;
  if (!sparse) {
    throw python_error();
  }
  // steals a reference to sparse
  if (PyModule_AddObject(module, "_sparse", sparse) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace smith::autograd
