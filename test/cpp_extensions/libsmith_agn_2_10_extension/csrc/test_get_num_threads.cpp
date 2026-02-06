#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>

uint32_t test_get_num_threads() {
  return smith::stable::get_num_threads();
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("test_get_num_threads() -> int");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_get_num_threads", SMITH_BOX(&test_get_num_threads));
}
