#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include "smith/csrc/autograd/VariableTypeUtils.h"
#include "smith/csrc/autograd/generated/ViewFuncs.h"

#include <smith/library.h>
#include <ATen/FunctionalInverses.h>
#include <ATen/FunctionalTensorWrapper.h>

// ${generated_comment}

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
$ops_headers
#endif

using namespace at;
using smith::autograd::CreationMeta;
using smith::autograd::as_view;
using smith::autograd::increment_version;

namespace smith {

namespace ADInplaceOrView {

namespace {
${inplace_or_view_method_definitions}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

SMITH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  ${inplace_or_view_wrapper_registrations};
}

}  // namespace
} // namespace smith
