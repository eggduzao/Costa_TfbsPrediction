#pragma once
#include <smith/csrc/Export.h>
#include <smith/csrc/lazy/core/ir_metadata.h>
#include <optional>
#include <vector>

namespace smith::lazy {

std::optional<SourceLocation> SMITH_PYTHON_API GetPythonFrameTop();

std::vector<SourceLocation> SMITH_PYTHON_API GetPythonFrames();

} // namespace smith::lazy
