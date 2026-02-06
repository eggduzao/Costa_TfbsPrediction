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

enum blacksmith_qnnp_status blacksmith_qnnp_create_fully_connected_sparse_dq_nc_q8(
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const void* kernel_col_indices,
    const void* kernel_row_values,
    const uint8_t* kernel_values,
    const uint32_t kernel_row_block_size,
    const uint32_t kernel_col_block_size,
    enum blacksmith_qnnp_sparse_matrix_indices_dtype kernel_indices_dtype,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool use_prepack_kernel,
    blacksmith_qnnp_operator_t* fully_connected_out) {
  blacksmith_qnnp_operator_t fully_connected = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_fully_connected_sparse_dq_nc_q8 failed because QNNPACK is not properly initialized");
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

  if (kernel_row_block_size == 8 && kernel_col_block_size == 1) {
    // This is to gate 8x1 on SSE2 since we have not implemented SSE2
    // kernel that supports 8x1 sparsity pattern.
    if (blacksmith_qnnp_params.q8gemm_sparse_c8x1.packA == NULL) {
      status = blacksmith_qnnp_status_invalid_parameter;
      goto error;
    }
  }

  fully_connected->sparse_matrix.indices_dtype = kernel_indices_dtype;
  switch (kernel_indices_dtype) {
    case blacksmith_qnnp_sparse_matrix_indices_dtype_uint32_t:
      fully_connected->sparse_matrix.col_indices_w32 =
          (const uint32_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w32 =
          (const uint32_t*)kernel_row_values;
      break;
    case blacksmith_qnnp_sparse_matrix_indices_dtype_uint16_t:
      fully_connected->sparse_matrix.col_indices_w16 =
          (const uint16_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w16 =
          (const uint16_t*)kernel_row_values;
      break;
    case blacksmith_qnnp_sparse_matrix_indices_dtype_uint8_t:
      fully_connected->sparse_matrix.col_indices_w8 =
          (const uint8_t*)kernel_col_indices;
      fully_connected->sparse_matrix.row_values_w8 =
          (const uint8_t*)kernel_row_values;
      break;
    case blacksmith_qnnp_sparse_matrix_indices_dtype_invalid:
      status = blacksmith_qnnp_status_invalid_parameter;
      blacksmith_qnnp_log_error(
          "Invalid indices dtype specified for qnnpack fully connected sparse");
      goto error;
  }

  fully_connected->sparse_matrix.values = kernel_values;
  fully_connected->sparse_matrix.row_block_size = kernel_row_block_size;
  fully_connected->sparse_matrix.col_block_size = kernel_col_block_size;

  fully_connected->groups = 1;
  fully_connected->group_input_channels = input_channels;
  fully_connected->group_output_channels = output_channels;

  fully_connected->kernel_zero_point = kernel_zero_points[0];

  fully_connected->dynamic_conv_quantization_params.input_zero_point =
    input_zero_point;
  fully_connected->dynamic_conv_quantization_params.kernel_zero_points =
    kernel_zero_points;
  fully_connected->dynamic_conv_quantization_params.multipliers =
    requantization_scales;

  // Always use prepacking based kernel
  fully_connected->ukernel_type = blacksmith_qnnp_ukernel_type_gemm_prepackA_sparse_dq;
  fully_connected->format = blacksmith_qnnp_format_quint8;

  *fully_connected_out = fully_connected;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(fully_connected);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_fully_connected_sparse_dq_nc_q8(
    blacksmith_qnnp_operator_t fully_connected,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    const float* bias,
    float* output,
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

  fully_connected->bias = bias;

  fully_connected->output_height = batch_size;
  fully_connected->output_width = 1;
  fully_connected->output = output;
  fully_connected->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
