#pragma once

#include <smith/headeronly/macros/Macros.h>

#include <sstream>
#include <stdexcept>

#define SMITH_SUCCESS 0
#define SMITH_FAILURE 1

HIDDEN_NAMESPACE_BEGIN(smith, headeronly, detail)
[[maybe_unused]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  throw std::runtime_error(ss.str());
}
HIDDEN_NAMESPACE_END(smith, headeronly, detail)

// This API is 100% inspired by AOTI_SMITH_ERROR_CODE_CHECK defined in
// blacksmith/smith/csrc/inductor/aoti_runtime/utils.h to handle the returns
// of the APIs in the shim. We are genericizing this for more global use
// of the shim beyond AOTI, for examples, see smith/csrc/stable/ops.h.
#define SMITH_ERROR_CODE_CHECK(call)                                       \
  if ((call) != SMITH_SUCCESS) {                                           \
    smith::headeronly::detail::throw_exception(#call, __FILE__, __LINE__); \
  }
