#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

SMITH_HIP_CPP_API miopenHandle_t getMiopenHandle();
} // namespace at::native
