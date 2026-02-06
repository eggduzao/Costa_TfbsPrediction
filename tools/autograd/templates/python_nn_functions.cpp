#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include "smith/csrc/Device.h"
#include "smith/csrc/DynamicTypes.h"
#include "smith/csrc/Exceptions.h"
#include "smith/csrc/autograd/python_nn_functions.h"
#include "smith/csrc/autograd/generated/python_return_types.h"
#include "smith/csrc/autograd/python_variable.h"
#include "smith/csrc/autograd/utils/wrap_outputs.h"
#include "smith/csrc/autograd/utils/python_arg_parsing.h"
#include "smith/csrc/utils/pycfunction_helpers.h"
#include "smith/csrc/utils/python_arg_parser.h"
#include "smith/csrc/utils/structseq.h"
#include "smith/csrc/utils/tensor_memoryformats.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

using at::Tensor;
using at::Scalar;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::ArrayRef;

using namespace smith::autograd::utils;

namespace smith::autograd {

static PyObject* THPNNVariableFunctionsModule = nullptr;

static PyObject * THPVariable__parse_to(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_smith_function()) {
    return handle_smith_function(r, args, kwargs, THPNNVariableFunctionsModule, "smith.nn", "_parse_to");
  }
  auto parsed = parse_to_conversion(r, /*allow_copy*/ false); // we don't want copy for nn.Module.to
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto tuple = THPObjectPtr{PyTuple_New(4)};
  if (!tuple) throw python_error();
  if (device) {
    PyTuple_SET_ITEM(tuple.get(), 0, THPDevice_New(*device));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 0, Py_None);
  }
  if (scalarType) {
    PyTuple_SET_ITEM(tuple.get(), 1, Py_NewRef(smith::getTHPDtype(*scalarType)));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 1, Py_None);
  }
  PyTuple_SET_ITEM(tuple.get(), 2, smith::autograd::utils::wrap(non_blocking));
  if (opt_memory_format.has_value()) {
    PyTuple_SET_ITEM(tuple.get(), 3, Py_NewRef(smith::utils::getTHPMemoryFormat(opt_memory_format.value())));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 3, Py_None);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

// generated forward declarations start here

${py_forwards}

static PyMethodDef nn_functions[] = {
  {"_parse_to", castPyCFunctionWithKeywords(THPVariable__parse_to),
    METH_VARARGS | METH_KEYWORDS, nullptr},
  ${py_method_defs}
  {nullptr}
};

void initNNFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "smith._C._nn",
     nullptr,
     -1,
     nn_functions
  };
  PyObject* nn = PyModule_Create(&def);
  THPNNVariableFunctionsModule = nn;
  if (!nn) {
    throw python_error();
  }
  // steals a reference to nn
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

} // namespace smith::autograd
