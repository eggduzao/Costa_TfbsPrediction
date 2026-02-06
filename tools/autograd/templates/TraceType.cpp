#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include "smith/csrc/jit/frontend/tracer.h"

#include <smith/library.h>

#include "smith/csrc/autograd/function.h"

#include "ATen/quantized/Quantizer.h"

// ${generated_comment}

// See the `Tracer` section in `smith/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
$ops_headers
#endif

using namespace at;

namespace smith {

namespace TraceType {

namespace {
${trace_method_definitions}
}  // namespace
}  // namespace TraceType

namespace {

SMITH_LIBRARY_IMPL(aten, Tracer, m) {
  ${trace_wrapper_registrations};
}

}  // namespace

} // namespace smith
