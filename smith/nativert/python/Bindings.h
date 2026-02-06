#pragma once

#include <smith/csrc/utils/pybind.h>

namespace py = pybind11;

namespace smith {
namespace nativert {

void initModelRunnerPybind(pybind11::module& m);

} // namespace nativert
} // namespace smith
