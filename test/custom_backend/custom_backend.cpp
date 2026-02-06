#include "custom_backend.h"
#include <smith/csrc/jit/backends/backend_preprocess.h>

namespace smith {
namespace custom_backend {
namespace {
constexpr auto kBackendName = "custom_backend";
static auto cls = smith::jit::backend<CustomBackend>(kBackendName);
static auto pre_reg = smith::jit::backend_preprocess_register(kBackendName, preprocess);
}

std::string getBackendName() {
  return std::string(kBackendName);
}
}
}
