#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

// Test contiguous with default memory format
Tensor my_contiguous(Tensor self) {
  return smith::stable::contiguous(self);
}

// Test contiguous with specified memory format
Tensor my_contiguous_memory_format(
    Tensor self,
    smith::headeronly::MemoryFormat memory_format) {
  return smith::stable::contiguous(self, memory_format);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_contiguous(Tensor self) -> Tensor");
  m.def("my_contiguous_memory_format(Tensor self, MemoryFormat memory_format) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_contiguous", SMITH_BOX(&my_contiguous));
  m.impl("my_contiguous_memory_format", SMITH_BOX(&my_contiguous_memory_format));
}
