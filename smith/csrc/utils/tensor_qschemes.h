#pragma once
#include <smith/csrc/QScheme.h>

namespace smith::utils {

PyObject* getTHPQScheme(at::QScheme qscheme);
void initializeQSchemes();

} // namespace smith::utils
