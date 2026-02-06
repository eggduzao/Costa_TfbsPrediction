#pragma once

#include <c10/macros/Macros.h>

namespace smith::jit {

SMITH_API double strtod_c(const char* nptr, char** endptr);
SMITH_API float strtof_c(const char* nptr, char** endptr);

} // namespace smith::jit
