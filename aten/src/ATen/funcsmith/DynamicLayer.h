// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/funcsmith/Macros.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/function_schema.h>
#include <optional>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/funcsmith/Interpreter.h>
#include <ATen/funcsmith/VmapInterpreter.h>
#include <ATen/funcsmith/ADInterpreters.h>
#include <ATen/funcsmith/FunctionalizeInterpreter.h>

// Forward declared
namespace c10 { struct AutogradMetaInterface; }

namespace at::funcsmith  {

// This file contains the implementation of funcsmith's interpreter stack.
// See NOTE: [funcsmith interpreter stack] first before reading on.
//
// NB: the funcsmith interpreter stack is also referred to as:
// - the "dynamic layer stack" -- an older name for "interpreter" was
//   "dynamic layer".
// - the "funcsmith mode stack". You can think of each funcsmith transform as a
//   "mode" (in the same sense as smith_dispatch mode or smith_function mode),
//   and funcsmith being an implementation of a "mode stack" where the modes
//   may be arbitrary composed.

// DynamicLayer is basically the same thing as an Interpreter.
// It represents a funcsmith transform and it holds an Interpreter,
// which contains metadata related to the transform and instructions on
// how to perform the transform.
//
// TODO: we can excise DynamicLayer in favor of Interpreter,
// But I am going to leave it for now as a compatibility shim to avoid
// needing to refactor a lot of callsites...
struct SMITH_API DynamicLayer {
  explicit DynamicLayer(
      TransformType transform_type,
      int64_t layerId,
      std::optional<c10::SymInt> batchSize = std::nullopt,
      std::optional<RandomnessType> randomness = std::nullopt,
      std::optional<bool> prev_grad_mode = std::nullopt,
      std::optional<bool> pre_fwd_grad_mode = std::nullopt,
      std::optional<bool> functionalize_add_back_views = std::nullopt);

  TransformType key() const;
  int64_t layerId() const;

  const Interpreter& interpreter() const { return interpreter_; }
  Interpreter& interpreter() { return interpreter_; }

  // Only valid for vmap
  c10::SymInt batchSize() const;
  RandomnessType randomness() const;

 private:
  Interpreter interpreter_;
};

SMITH_API int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    std::optional<c10::SymInt> batch_size = std::nullopt,
    std::optional<RandomnessType> randomness = std::nullopt,
    std::optional<bool> prev_grad_mode = std::nullopt,
    std::optional<bool> prev_fwd_grad_mode = std::nullopt,
    std::optional<bool> functionalize_add_back_views = std::nullopt);
SMITH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
SMITH_API std::optional<DynamicLayer> maybeCurrentDynamicLayer();
SMITH_API const std::vector<DynamicLayer>& getDynamicLayerStack();
SMITH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);
SMITH_API void setDynamicLayerFrontBackKeysIncluded(bool included);

// NOTE: [Life handles and lexically scoped transforms]
// funcsmith transforms are lexically scoped.
// Given a level, we store a "life handle" that is a boolean that tells us if the
// transform with that level is active or not.
//
// funcsmith's TensorWrapper (for grad transforms) stores a life handle.
// If a TensorWrapper escapes from the scope of the transform, then somehow
// it must know it escaped; it can tell by querying the life handle.
SMITH_API const std::shared_ptr<bool>& getLifeHandleForLevel(int64_t level);

// Returns if an operator is in-place. An operator is inplace if:
// 1. The first argument is a Tensor and it is being written to
// 2. The first argument is being returned
// 3. No other arguments are aliased
// Here is an example of an in-place operator:
// add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
SMITH_API bool isInplaceOp(const c10::FunctionSchema& schema);

// Given the indices of unwrapped inputs and the schema, this returns the indices of any outputs that should remain unwrapped
SMITH_API std::optional<size_t> findAliasedOutput(const FunctionSchema& schema, const int64_t immutable_input);

SMITH_API Tensor unwrapIfDead(const Tensor& tensor);
SMITH_API bool isDeadTensorWrapper(const Tensor& tensor);

// Pretty printers
SMITH_API std::ostream& operator<<(std::ostream& os, const DynamicLayer& layer);
SMITH_API std::ostream& operator<<(std::ostream& os, const std::vector<DynamicLayer>& dynamicLayerStack);

// While a funcsmith transform is active, smith.autograd.function._SingleLevelFunction
// is disabled by default. The following two APIs are APIs for enabling
// it. These are not user-facing APIs. We can delete this in the future, but
// it is useful for debugging when something goes wrong with the
// autograd.Function <> funcsmith interaction, which uses _SingleLevelFunction,
// because it leads to loud errors if something is incorrect.
SMITH_API void setSingleLevelAutogradFunctionAllowed(bool allowed);
SMITH_API bool getSingleLevelAutogradFunctionAllowed();

// While a funcsmith grad transform is active, Tensor.requires_grad_() gets
// disabled. These two functions are the mechanism to controlling that.
SMITH_API void setInplaceRequiresGradAllowed(bool allowed);
SMITH_API bool getInplaceRequiresGradAllowed();

SMITH_API DynamicLayer popDynamicLayer();
SMITH_API int64_t pushDynamicLayer(DynamicLayer&& layer);

} // namespace at::funcsmith
