#pragma once

#include <c10/macros/Export.h>

namespace smith::verbose {
SMITH_API int _mkl_set_verbose(int enable);
SMITH_API int _mkldnn_set_verbose(int level);
} // namespace smith::verbose
