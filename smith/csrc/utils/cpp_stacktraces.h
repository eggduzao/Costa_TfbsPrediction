#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/profiler/unwind/unwind.h>

namespace smith {
SMITH_API bool get_cpp_stacktraces_enabled();
SMITH_API smith::unwind::Mode get_symbolize_mode();
} // namespace smith
