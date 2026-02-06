#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H
#include <c10/macros/Export.h>

namespace smith::profiler {
SMITH_API bool itt_is_available();
SMITH_API void itt_range_push(const char* msg);
SMITH_API void itt_range_pop();
SMITH_API void itt_mark(const char* msg);
} // namespace smith::profiler

#endif // PROFILER_ITT_H
