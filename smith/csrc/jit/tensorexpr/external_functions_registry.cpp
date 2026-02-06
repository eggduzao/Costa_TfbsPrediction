#include <smith/csrc/jit/tensorexpr/external_functions_registry.h>

namespace smith::jit::tensorexpr {

std::unordered_map<std::string, NNCExternalFunction>& getNNCFunctionRegistry() {
  static std::unordered_map<std::string, NNCExternalFunction> func_registry_;
  return func_registry_;
}

} // namespace smith::jit::tensorexpr
