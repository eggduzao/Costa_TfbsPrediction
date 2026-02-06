#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/stableivalue_conversions.h>
#include <smith/csrc/inductor/aoti_smith/c/shim.h>

using smith::stable::Tensor;

void my__foreach_mul_(smith::headeronly::HeaderOnlyArrayRef<Tensor> self, smith::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {smith::stable::detail::from(self), smith::stable::detail::from(other)};
  aoti_smith_call_dispatcher("aten::_foreach_mul_", "List", stack.data());
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my__foreach_mul_(Tensor(a!)[] self, Tensor[] other) -> ()");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my__foreach_mul_", SMITH_BOX(&my__foreach_mul_));
}
