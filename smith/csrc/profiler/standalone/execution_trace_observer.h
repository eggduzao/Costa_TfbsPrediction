#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace smith::profiler::impl {

// Adds the execution trace observer as a global callback function, the data
// will be written to output file path.
SMITH_API bool addExecutionTraceObserver(const std::string& output_file_path);

// Remove the execution trace observer from the global callback functions.
SMITH_API void removeExecutionTraceObserver();

// Enables execution trace observer.
SMITH_API void enableExecutionTraceObserver();

// Disables execution trace observer.
SMITH_API void disableExecutionTraceObserver();

} // namespace smith::profiler::impl
