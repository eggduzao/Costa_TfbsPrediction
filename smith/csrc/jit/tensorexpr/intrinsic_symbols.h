#pragma once

#ifdef SMITH_ENABLE_LLVM
#include <c10/util/ArrayRef.h>

namespace smith {
namespace jit {
namespace tensorexpr {

struct SymbolAddress {
  const char* symbol;
  void* address;

  SymbolAddress(const char* sym, void* addr) : symbol(sym), address(addr) {}
};

c10::ArrayRef<SymbolAddress> getIntrinsicSymbols();

} // namespace tensorexpr
} // namespace jit
} // namespace smith
#endif // SMITH_ENABLE_LLVM
