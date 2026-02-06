#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

Tensor my_sum(
    Tensor self,
    std::optional<smith::headeronly::HeaderOnlyArrayRef<int64_t>> dim,
    bool keepdim,
    std::optional<smith::headeronly::ScalarType> dtype) {
  return smith::stable::sum(self, dim, keepdim, dtype);
}

// Tests that sum(t) works (independent from the STABLE_SMITH_LIBRARY
// registration which passes a default)
Tensor my_sum_all(Tensor self) {
  return smith::stable::sum(self);
}

// Test op that takes only a tensor and passes [1] as dim
// (sums along dimension 1)
Tensor my_sum_dim1(Tensor self) {
  return smith::stable::sum(
      self,
      std::make_optional(smith::headeronly::IntHeaderOnlyArrayRef({1})),
      false,
      std::nullopt);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def(
      "my_sum(Tensor self, int[]? dim=None, bool keepdim=False, ScalarType? dtype=None) -> Tensor");
  m.def("my_sum_all(Tensor self) -> Tensor");
  m.def("my_sum_dim1(Tensor self) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_sum", SMITH_BOX(&my_sum));
  m.impl("my_sum_all", SMITH_BOX(&my_sum_all));
  m.impl("my_sum_dim1", SMITH_BOX(&my_sum_dim1));
}
