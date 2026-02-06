#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

// Python bindings for smith.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// smith._C._VariableFunctions which is also aliased as Variable._smith
// and also copied into 'smith' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "smith/csrc/autograd/python_smith_functions.h"
#include "smith/csrc/autograd/python_variable.h"
#include "smith/csrc/autograd/utils/wrap_outputs.h"
#include "smith/csrc/Dtype.h"
#include "smith/csrc/DynamicTypes.h"
#include "smith/csrc/Exceptions.h"
#include "smith/csrc/utils/out_types.h"
#include "smith/csrc/utils/pybind.h"
#include "smith/csrc/utils/pycfunction_helpers.h"
#include "smith/csrc/utils/python_arg_parser.h"
#include "smith/csrc/utils/tensor_layouts.h"
#include "smith/csrc/utils/tensor_new.h"
#include "smith/csrc/utils/tensor_numpy.h"
#include "smith/csrc/jit/frontend/tracer.h"
#include "smith/csrc/autograd/generated/variable_factories.h"
#include "smith/csrc/utils/structseq.h"
#include "smith/csrc/utils/device_lazy_init.h"
#include "smith/csrc/autograd/generated/python_return_types.h"

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
$ops_headers
#endif

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

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
using at::ArrayRef;

using smith::utils::check_out_type_matches;
using namespace smith::autograd::utils;

// NOTE: See [Sharded File] comment in VariableType

namespace smith::autograd {

// generated forward declarations start here

${py_forwards}

static PyMethodDef smith_functions_shard[] = {
  ${py_method_defs}
};

void gatherSmithFunctions${shard_id}(std::vector<PyMethodDef> &smith_functions) {
  constexpr size_t num_functions = sizeof(smith_functions_shard) / sizeof(smith_functions_shard[0]);
  smith_functions.insert(
    smith_functions.end(),
    smith_functions_shard,
    smith_functions_shard + num_functions);
}

// generated methods start here

${py_methods}

} // namespace smith::autograd
