/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <blacksmith_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

enum blacksmith_qnnp_status blacksmith_qnnp_create_fully_connected_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    blacksmith_qnnp_operator_t* fully_connected_out) {
  blacksmith_qnnp_operator_t fully_connected = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_unsupported_parameter;

  for (int i = 0; i < output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !isnormal(requantization_scales[i])) {
      blacksmith_qnnp_log_error(
          "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
          requantization_scales[i]);
      goto error;
    }
  }

  status = blacksmith_qnnp_status_out_of_memory;

  fully_connected = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (fully_connected == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  const uint32_t nr = blacksmith_qnnp_params.q8conv.nr;
  const uint32_t kr = blacksmith_qnnp_params.q8conv.kr;

  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  fully_connected->packed_weights =
      malloc(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (fully_connected->packed_weights == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    goto error;
  }
  memset(
      fully_connected->packed_weights,
      kernel_zero_points[0],
      n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));

  blacksmith_pack_q8gemm_w(
      output_channels,
      input_channels,
      nr,
      nr,
      kr,
#if !BLACKSMITH_QNNPACK_RUNTIME_QUANTIZATION
      input_zero_point,
      kernel_zero_points[0],
#endif
      kernel,
      bias,
#if BLACKSMITH_QNNPACK_RUNTIME_QUANTIZATION
      kernel_zero_points,
#endif
      fully_connected->packed_weights);

  fully_connected->groups = 1;
  fully_connected->group_input_channels = input_channels;
  fully_connected->group_output_channels = output_channels;

  fully_connected->kernel_zero_point = kernel_zero_points[0];

  fully_connected->conv_quantization_params =
      blacksmith_qnnp_compute_conv_quantization_params(
          input_zero_point,
          kernel_zero_points,
          requantization_scales,
          output_zero_point,
          output_min,
          output_max);

  fully_connected->ukernel_type = blacksmith_qnnp_ukernel_type_gemm;
  fully_connected->format = blacksmith_qnnp_format_quint8;

  *fully_connected_out = fully_connected;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(fully_connected);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_fully_connected_nc_q8(
    blacksmith_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    fully_connected->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  fully_connected->batch_size = 1;
  fully_connected->input_height = batch_size;
  fully_connected->input_width = 1;
  fully_connected->input = input;
  fully_connected->input_pixel_stride = input_stride;

  fully_connected->output_height = batch_size;
  fully_connected->output_width = 1;
  fully_connected->output = output;
  fully_connected->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
