#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::detail {
SMITH_PYTHON_API py::handle getSmithApiFunction(const c10::OperatorHandle& op);
}

// TODO: Move these to a proper namespace
SMITH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
SMITH_PYTHON_API void initializeGlobalPyInterpreter();
