#include <c10/macros/Macros.h>
#include <smith/csrc/jit/backends/backend_debug_info.h>

namespace smith::jit::backend {
namespace {
#ifdef BUILD_LITE_INTERPRETER
static auto cls = smith::class_<BlacksmithBackendDebugInfoDummy>(
                      kBackendUtilsNamespace,
                      kBackendDebugInfoClass)
                      .def(smith::init<>());
#else
static auto cls = smith::class_<BlacksmithBackendDebugInfo>(
                      kBackendUtilsNamespace,
                      kBackendDebugInfoClass)
                      .def(smith::init<>());
#endif

} // namespace
} // namespace smith::jit::backend
