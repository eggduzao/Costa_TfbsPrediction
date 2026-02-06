// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/ATen.h>

#include <typeinfo>

inline std::string smith_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

#define TENSOR_NDIM_EQUALS(ten, dims)      \
  SMITH_CHECK(                             \
      (ten).ndimension() == (dims),        \
      "Tensor '" #ten "' must have " #dims \
      " dimension(s). "                    \
      "Found ",                            \
      (ten).ndimension())

#define TENSOR_ON_CPU(x)                                      \
  SMITH_CHECK(                                                \
      !x.is_cuda(),                                           \
      #x " must be a CPU tensor; it is currently on device ", \
      smith_tensor_device_name(x))

#define TENSOR_ON_CUDA_GPU(x)                                  \
  SMITH_CHECK(                                                 \
      x.is_cuda(),                                             \
      #x " must be a CUDA tensor; it is currently on device ", \
      smith_tensor_device_name(x))
