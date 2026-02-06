#pragma once

#include <c10/core/MemoryFormat.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/utils/python_stub.h>

namespace smith::utils {

void initializeMemoryFormats();

// This methods returns a borrowed reference!
SMITH_PYTHON_API PyObject* getTHPMemoryFormat(
    c10::MemoryFormat /*memory_format*/);

} // namespace smith::utils
