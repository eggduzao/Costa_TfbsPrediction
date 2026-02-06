#pragma once

#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/onnx/onnx.h>

namespace smith::autograd {

struct SymbolicContext {
  jit::Block* block;
};

struct symbolic_unconvertible : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace smith::autograd
