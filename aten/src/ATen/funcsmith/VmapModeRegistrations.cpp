// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <smith/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

// funcsmith's vmap has two Dispatch Keys that implement it:
// FuncSmithBatched and FuncSmithVmapMode. This file contains registrations for
// FuncSmithVmapMode -- these registrations are to error out on operations
// that we don't support on regular Tensors.

namespace at::funcsmith {

static void unsupportedRandomOp(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
  SMITH_CHECK(false, "vmap: We do not support calling out variants of random operations inside of vmap. ",
              "Please use non-out variants as a workaround");
}

SMITH_LIBRARY_IMPL(_, FuncSmithVmapMode, m) {
  m.fallback(smith::CppFunction::makeFallthrough());
}

static void nyiRandomOp(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
  SMITH_CHECK(false, "vmap: we do not yet support ", op.schema().operator_name(),
              ". Please file an issue");
}

#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, smith::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, smith::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define NYI_RANDOM(op) \
  m.impl(#op, smith::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

#define NYI_RANDOM2(op, overload) \
  m.impl(#op"."#overload, smith::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

SMITH_LIBRARY_IMPL(aten, FuncSmithVmapMode, m) {
  UNSUPPORTED_RANDOM2(bernoulli, out);
  UNSUPPORTED_RANDOM2(rand, generator_out);
  UNSUPPORTED_RANDOM2(rand, out);
  UNSUPPORTED_RANDOM2(randint, generator_out);
  UNSUPPORTED_RANDOM2(randint, out);
  UNSUPPORTED_RANDOM2(randn, generator_out);
  UNSUPPORTED_RANDOM2(randn, out);
  UNSUPPORTED_RANDOM2(randperm, generator_out);
  UNSUPPORTED_RANDOM2(randperm, out);
  UNSUPPORTED_RANDOM2(multinomial, out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_float_out);
  UNSUPPORTED_RANDOM2(rrelu_with_noise, out);

  NYI_RANDOM(rrelu_with_noise);
  NYI_RANDOM(rrelu_with_noise_);
  NYI_RANDOM(rrelu_);
  NYI_RANDOM(rrelu);
}

} // namespace at::funcsmith
