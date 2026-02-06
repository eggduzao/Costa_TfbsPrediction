#include <ATen/FuncSmithTLS.h>

namespace at::funcsmith {

namespace {

thread_local std::unique_ptr<FuncSmithTLSBase> kFuncSmithTLS = nullptr;

}

std::unique_ptr<FuncSmithTLSBase> getCopyOfFuncSmithTLS() {
  if (kFuncSmithTLS == nullptr) {
    return nullptr;
  }
  return kFuncSmithTLS->deepcopy();
}

void setFuncSmithTLS(const std::shared_ptr<const FuncSmithTLSBase>& state) {
  if (state == nullptr) {
    kFuncSmithTLS = nullptr;
    return;
  }
  kFuncSmithTLS = state->deepcopy();
}

std::unique_ptr<FuncSmithTLSBase>& funcsmithTLSAccessor() {
  return kFuncSmithTLS;
}


} // namespace at::funcsmith
