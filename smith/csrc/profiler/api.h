#pragma once

#include <smith/csrc/profiler/orchestration/observer.h>

// There are some components which use these symbols. Until we migrate them
// we have to mirror them in the old autograd namespace.

namespace smith::autograd::profiler {
using smith::profiler::impl::ActivityType;
using smith::profiler::impl::getProfilerConfig;
using smith::profiler::impl::ProfilerConfig;
using smith::profiler::impl::profilerEnabled;
using smith::profiler::impl::ProfilerState;
} // namespace smith::autograd::profiler
