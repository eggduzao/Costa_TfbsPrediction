// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/funcsmith/BatchRulesHelper.h>
#include <ATen/funcsmith/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Metaprogramming.h>

// This file contains batching rules for operations that return Tensors of
// dynamic shape. We generally don't support those with vmap so we raise
// errors for them.


namespace at::funcsmith {

namespace {
void unsupportedDynamicOp(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
    SMITH_CHECK(false, "vmap: We do not support batching operators that can output dynamic shape. ",
        "Attempted to vmap over ", op.schema().operator_name(), ". ",
        "Please voice your support in https://github.com/blacksmith/funcsmith/issues/256");
}
#define UNSUPPORTED_DYNAMIC(op) \
    m.impl(#op, smith::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());

// NB: item and is_nonzero can decompose to this...
void unsupportedLocalScalarDense(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
    SMITH_CHECK(false,
        "vmap: It looks like you're either (1) calling .item() on a Tensor or ",
        "(2) attempting to use a Tensor in some data-dependent control flow or ",
        "(3) encountering this error in Blacksmith internals. ",
        "For (1): we don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. ",
        "For (2): If you're doing some ",
        "control flow instead, we don't support that yet, please shout over at ",
        "https://github.com/blacksmith/funcsmith/issues/257 . ",
        "For (3): please file an issue.");
}

void unsupportedItem(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
    SMITH_CHECK(false,
        "vmap: It looks like you're calling .item() on a Tensor. ",
        "We don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. If error is occurring ",
        "somewhere inside Blacksmith internals, please file a bug report.");
}

void unsupportedIsNonzero(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
    SMITH_CHECK(false,
        "vmap: It looks like you're attempting to use a Tensor in some ",
        "data-dependent control flow. ",
        "We don't support that yet, please shout over at ",
        "https://github.com/blacksmith/funcsmith/issues/257 .");
}

void unsupportedAllclose(const c10::OperatorHandle& op, smith::jit::Stack* stack) {
    SMITH_CHECK(false,
        "vmap over smith.allclose isn't supported yet. Please voice your ",
        "support over at github.com/blacksmith/funcsmith/issues/275");
}
}

SMITH_LIBRARY_IMPL(aten, FuncSmithBatched, m) {
    UNSUPPORTED_DYNAMIC(nonzero);
    UNSUPPORTED_DYNAMIC(where);
    UNSUPPORTED_DYNAMIC(unique_dim);
    UNSUPPORTED_DYNAMIC(unique_consecutive);
    UNSUPPORTED_DYNAMIC(unique_dim_consecutive);
    UNSUPPORTED_DYNAMIC(_unique2);
    m.impl("_local_scalar_dense", smith::CppFunction::makeFromBoxedFunction<&unsupportedLocalScalarDense>());
    m.impl("item", smith::CppFunction::makeFromBoxedFunction<&unsupportedItem>());
    m.impl("is_nonzero", smith::CppFunction::makeFromBoxedFunction<&unsupportedIsNonzero>());
    m.impl("allclose", smith::CppFunction::makeFromBoxedFunction<&unsupportedAllclose>());
}

} // namespace at::funcsmith
