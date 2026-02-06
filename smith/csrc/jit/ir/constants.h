#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/frontend/source_range.h>
#include <smith/csrc/jit/ir/scope.h>

// helpers for handling constants in the IR
// - create constant nodes from ints, floats, complex, intlist, Tensors, and
// other types
// - implement primitive constant ops.

namespace smith::jit {

using ::c10::IValue;

struct Graph;
struct Value;

// thrown when insertConstant cannot encode the IValue into a graph
struct SMITH_API constant_not_supported_error : public std::runtime_error {
  using runtime_error::runtime_error;
};

SMITH_API Value* insertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc = std::nullopt,
    std::optional<ScopePtr> scope = std::nullopt);

// note: prefer g.insertConsant(val, loc) which does exactly the same thing
// this function is only declared/defined here because its implementation is
// closely related to the implementation of prim::Constant that is also in
// constants.cpp.
//
// returns a std::nullopt if the IValue kind cannot be inserted as a constant
SMITH_API std::optional<Value*> tryInsertConstant(
    Graph& g,
    const IValue& val,
    std::optional<SourceRange> loc = std::nullopt,
    std::optional<ScopePtr> scope = std::nullopt);

////////////////////////////////////////////////////////////////////////////////
// Helper for retrieving constants
////////////////////////////////////////////////////////////////////////////////

// attempt to convert a (possibly constant) Value* into an interpreter value
// (IValue). returns std::nullopt if the Value* was not constant
SMITH_API std::optional<IValue> toIValue(const Value* v);

// if a value is a constant then try to turn into type T using the
// same rules as the interpreter
template <typename T>
std::optional<T> constant_as(const Value* v) {
  if (auto ivalue = toIValue(v)) {
    return ivalue->to<T>();
  }
  return std::nullopt;
}
} // namespace smith::jit
