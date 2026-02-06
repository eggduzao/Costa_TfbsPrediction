#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>

using smith::stable::Tensor;

Tensor my_sum_out(
    Tensor out,
    Tensor self,
    std::optional<smith::headeronly::HeaderOnlyArrayRef<int64_t>> dim,
    bool keepdim = false,
    std::optional<smith::headeronly::ScalarType> dtype = std::nullopt) {
  return sum_out(out, self, dim, keepdim, dtype);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_sum_out(Tensor(a!) out, Tensor self, int[]? dim=None, bool keepdim=False, ScalarType? dtype=None) -> Tensor(a!)");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_sum_out", SMITH_BOX(&my_sum_out));
}
