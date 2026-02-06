#include <smith/csrc/autograd/jit_decomp_interface.h>

namespace smith::autograd::impl {

namespace {
JitDecompInterface* impl = nullptr;
}

void setJitDecompImpl(JitDecompInterface* impl_) {
  impl = impl_;
}

JitDecompInterface* getJitDecompImpl() {
  return impl;
}

} // namespace smith::autograd::impl
