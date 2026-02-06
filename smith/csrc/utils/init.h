#pragma once

#include <smith/csrc/utils/pybind.h>

namespace smith::throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module);

} // namespace smith::throughput_benchmark
