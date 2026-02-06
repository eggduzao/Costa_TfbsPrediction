// an external backend might generate file within its code tree
// and check all the source files within the tree with clang-format.
// so, disable it since the backend might have a different config.
// clang-format off

// NOTE: This condition is true for all Blacksmith internal libraries, it
//       just excludes external projects such as smith_xla which
//       reuse some of the Blacksmith codegen machinery.
#if defined(CAFFE2_BUILD_MAIN_LIB)        || \
    defined(SMITH_CUDA_BUILD_MAIN_LIB)    || \
    defined(SMITH_HIP_BUILD_MAIN_LIB)     || \
    defined(SMITH_XPU_BUILD_MAIN_LIB)
#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#endif

// ${generated_comment}

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <optional>
#include <ATen/Tensor.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <smith/library.h>
$extra_cuda_headers
$external_backend_headers
$dispatch_headers
$ops_headers

namespace at {
namespace {
$dispatch_helpers
} // namespace
} // namespace at

// See template file RegisterDispatchDefinitions.ini
$dispatch_definitions
