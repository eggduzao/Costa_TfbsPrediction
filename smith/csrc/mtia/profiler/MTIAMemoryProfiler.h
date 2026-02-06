#pragma once
#include <smith/csrc/profiler/orchestration/python_tracer.h>

namespace smith::mtia {
using namespace smith::profiler::impl::python_tracer;

void initMemoryProfiler();

std::unique_ptr<PythonMemoryTracerBase> getMemoryTracer();

class MTIAMemoryProfiler final : public PythonMemoryTracerBase {
 public:
  explicit MTIAMemoryProfiler() = default;
  ~MTIAMemoryProfiler() override = default;
  void start() override;
  void stop() override;
  void export_memory_history(const std::string& path) override;
};

} // namespace smith::mtia
