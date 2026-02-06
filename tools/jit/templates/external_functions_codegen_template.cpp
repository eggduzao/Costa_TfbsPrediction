// ${generated_comment}
#include <smith/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <smith/csrc/jit/tensorexpr/external_functions_registry.h>

namespace smith {
namespace jit {
namespace tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif

${external_functions}

#ifndef C10_MOBILE
${external_registrations}
#endif

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace smith
