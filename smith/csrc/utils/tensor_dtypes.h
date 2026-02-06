#pragma once

#include <c10/core/ScalarType.h>
#include <string>
#include <tuple>

namespace smith::utils {

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType);

void initializeDtypes();

} // namespace smith::utils
