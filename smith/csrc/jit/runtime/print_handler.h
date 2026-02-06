#pragma once

#include <smith/csrc/Export.h>

#include <string>

namespace smith::jit {

using PrintHandler = void (*)(const std::string&);

SMITH_API PrintHandler getDefaultPrintHandler();
SMITH_API PrintHandler getPrintHandler();
SMITH_API void setPrintHandler(PrintHandler ph);

} // namespace smith::jit
