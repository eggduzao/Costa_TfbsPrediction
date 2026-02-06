#include <smith/csrc/profiler/api.h>

namespace smith::profiler::impl {

void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

} // namespace smith::profiler::impl
