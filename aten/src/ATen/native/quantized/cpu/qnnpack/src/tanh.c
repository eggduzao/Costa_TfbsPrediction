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

enum blacksmith_qnnp_status blacksmith_qnnp_create_tanh_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    blacksmith_qnnp_operator_t* tanh_out) {
  blacksmith_qnnp_operator_t tanh_op = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_tanh_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_invalid_parameter;

  if (channels == 0) {
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = blacksmith_qnnp_status_unsupported_parameter;

  if (output_scale != 0x2.0p-8f) {  // [-1, 1] range in 8 bits = 2.0 / 256
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with %.7g output scale: only output scale of 2/256 is supported",
        output_scale);
    goto error;
  }

  if (output_zero_point != 128) {
    blacksmith_qnnp_log_error(
        "failed to create TanH operator with %" PRIu8
        " output zero point: only output zero point of 128 is supported",
        output_zero_point);
    goto error;
  }

  status = blacksmith_qnnp_status_out_of_memory;

  tanh_op = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (tanh_op == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  tanh_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (tanh_op->lookup_table == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate 256 bytes for TanH lookup table");
    goto error;
  }

  uint8_t* lookup_table = tanh_op->lookup_table;
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  for (int32_t i = 0; i < 256; i++) {
    const float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    /* Scale tanh(x) by 1 / output scale = 128.0
       Also, offset by the zero_point from the scaled value, as we assume UINT8
    */
    float scaled_tanh_x = 128.0f * tanhf(x) + 128.0f;
    if (scaled_tanh_x < scaled_min) {
      scaled_tanh_x = scaled_min;
    }
    if (scaled_tanh_x > scaled_max) {
      scaled_tanh_x = scaled_max;
    }
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_tanh_x);
  }

  tanh_op->channels = channels;

  tanh_op->ukernel_type = blacksmith_qnnp_ukernel_type_lut;
  tanh_op->format = blacksmith_qnnp_format_quint8;

  *tanh_out = tanh_op;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(tanh_op);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_tanh_nc_q8(
    blacksmith_qnnp_operator_t tanh,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_tanh_nc_q8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    tanh->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  tanh->batch_size = batch_size;
  tanh->input = input;
  tanh->input_pixel_stride = input_stride;
  tanh->output = output;
  tanh->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
