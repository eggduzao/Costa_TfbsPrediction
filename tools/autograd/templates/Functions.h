#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "smith/csrc/autograd/function.h"
#include "smith/csrc/autograd/variable.h"
#include "smith/csrc/autograd/saved_variable.h"
#include <smith/csrc/Export.h>

#include <c10/core/SymIntArrayRef.h>

namespace smith { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using std::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [&saved_for](const SavedVariable& x) {
    // TODO(crcrpar): Use `std::move(saved_for)` to avoid incrementing refcount, which would need refactoring.
    return static_cast<Tensor>(x.unpack(saved_for));
  });
}

inline c10::List<std::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  smith::List<std::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    auto var = v.unpack(saved_for);
    result.push_back(var.defined() ? std::optional<Tensor>(var) : ::std::nullopt);
  }
  return result;
}

using smith::autograd::TypeAndSize;

${autograd_function_declarations}

}}} // namespace smith::autograd::generated
