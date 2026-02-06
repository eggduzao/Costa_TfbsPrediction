// This file defines classes for registering standard lowerings from JIT to TE
// IR.
#pragma once

#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/runtime/interpreter.h>
#include <smith/csrc/jit/tensorexpr/analysis.h>
#include <smith/csrc/jit/tensorexpr/codegen.h>
#include <smith/csrc/jit/tensorexpr/tensor.h>

namespace smith::jit::tensorexpr {

using ArgNone = std::monostate;
using BufList = std::vector<tensorexpr::BufHandle>;
using DoubleList = std::vector<double>;
using IntList = std::vector<int64_t>;
using ArgValue = std::variant<
    tensorexpr::BufHandle,
    tensorexpr::VarHandle,
    double,
    int64_t,
    bool,
    BufList,
    DoubleList,
    IntList,
    std::string,
    ArgNone>;

using NNCLoweringFunction = std::function<Tensor(
    const std::vector<ArgValue>&,
    const std::vector<ExprHandle>&,
    const std::vector<ExprHandle>&,
    const std::optional<ScalarType>&,
    at::Device)>;

SMITH_API FunctionSchemaMap<NNCLoweringFunction>& getNNCLoweringRegistry();
SMITH_API NNCLoweringFunction getStandardLoweringFor(const std::string& op);

struct RegisterNNCLoweringsFunction {
  RegisterNNCLoweringsFunction(
      const std::vector<std::string>& schemas,
      const NNCLoweringFunction& fn);
};

} // namespace smith::jit::tensorexpr
