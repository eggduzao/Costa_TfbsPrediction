/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <blacksmith_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

enum blacksmith_qnnp_status blacksmith_qnnp_create_sigmoid_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    blacksmith_qnnp_operator_t* sigmoid_out) {
  blacksmith_qnnp_operator_t sigmoid_op = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_invalid_parameter;

  if (channels == 0) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = blacksmith_qnnp_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    blacksmith_qnnp_log_error(
        "failed to create Sigmoid operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = blacksmith_qnnp_status_out_of_memory;

  sigmoid_op = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (sigmoid_op == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  sigmoid_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (sigmoid_op->lookup_table == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate 256 bytes for Sigmoid lookup table");
    goto error;
  }

  uint8_t* lookup_table = sigmoid_op->lookup_table;
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  for (int32_t i = 0; i < 256; i++) {
    const float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    /* Scale sigmoid(x) by 1 / output scale = 256.0 */
    float scaled_sigmoid_x = 256.0f / (1.0f + expf(-x));
    if (scaled_sigmoid_x < scaled_min) {
      scaled_sigmoid_x = scaled_min;
    }
    if (scaled_sigmoid_x > scaled_max) {
      scaled_sigmoid_x = scaled_max;
    }
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_sigmoid_x);
  }

  sigmoid_op->channels = channels;

  sigmoid_op->ukernel_type = blacksmith_qnnp_ukernel_type_lut;
  sigmoid_op->format = blacksmith_qnnp_format_quint8;

  *sigmoid_out = sigmoid_op;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(sigmoid_op);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_sigmoid_nc_q8(
    blacksmith_qnnp_operator_t sigmoid,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    sigmoid->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  sigmoid->batch_size = batch_size;
  sigmoid->input = input;
  sigmoid->input_pixel_stride = input_stride;
  sigmoid->output = output;
  sigmoid->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
