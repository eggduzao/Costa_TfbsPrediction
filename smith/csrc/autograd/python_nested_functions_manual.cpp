#include <smith/csrc/autograd/python_nested_functions.h>
#include <smith/csrc/utils/nested.h>
#include <smith/csrc/utils/pycfunction_helpers.h>
#include <smith/csrc/utils/python_arg_parser.h>
#include <smith/smith.h>

namespace smith::autograd {

static PyObject* THPVariable_nested_tensor(
    PyObject* /*self*/,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
  });

  constexpr int ctor_num_args = 5;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  jit::tracer::warn(
      "smith.nested.nested_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(smith::utils::nested_tensor_ctor(
      smith::tensors::get_default_dispatch_key(),
      smith::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef nested_functions_manual[] = {
    {"nested_tensor",
     castPyCFunctionWithKeywords(THPVariable_nested_tensor),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
};

PyMethodDef* get_nested_functions_manual() {
  return nested_functions_manual;
}

} // namespace smith::autograd
