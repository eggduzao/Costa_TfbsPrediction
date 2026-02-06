#pragma once

#include <smith/csrc/lazy/core/ir.h>

namespace smith::lazy {

SMITH_API NodePtr operator+(const Value& node1, const Value& node2);
SMITH_API NodePtr operator-(const Value& node1, const Value& node2);
SMITH_API NodePtr operator*(const Value& node1, const Value& node2);
SMITH_API NodePtr operator/(const Value& node1, const Value& node2);

} // namespace smith::lazy
