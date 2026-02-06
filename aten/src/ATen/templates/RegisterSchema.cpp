// ${generated_comment}
#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <smith/library.h>

namespace at {
SMITH_LIBRARY(aten, m) {
  ${aten_schema_registrations};
  // Distributed Ops
  // Implementations located in smith/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
}
${schema_registrations}
}  // namespace at
