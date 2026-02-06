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

enum blacksmith_qnnp_status blacksmith_qnnp_create_hardsigmoid_nc_q8(
    size_t channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    blacksmith_qnnp_operator_t* hardsigmoid_out) {
  blacksmith_qnnp_operator_t hardsigmoid_op = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_hardsigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_invalid_parameter;

  if (channels == 0) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  if (output_min >= output_max) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = blacksmith_qnnp_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    blacksmith_qnnp_log_error(
        "failed to create Hardsigmoid operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = blacksmith_qnnp_status_out_of_memory;

  hardsigmoid_op = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (hardsigmoid_op == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  hardsigmoid_op->lookup_table = malloc(256 * sizeof(uint8_t));
  if (hardsigmoid_op->lookup_table == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate 256 bytes for Hardsigmoid lookup table");
    goto error;
  }

  uint8_t* lookup_table = hardsigmoid_op->lookup_table;
  const float scaled_min = (float)(int32_t)output_min;
  const float scaled_max = (float)(int32_t)output_max;
  const float inv_output_scale = 1.0f / output_scale;
  for (int32_t i = 0; i < 256; i++) {
    float x =
        input_scale * (float)(i - (int32_t)(uint32_t)input_zero_point);
    // hardsigmoid, no min/max functions in C
    float x2 = x + 3.0f;
    x2 = x2 > 0.0f ? x2 : 0.0f;
    x2 = x2 < 6.0f ? x2 : 6.0f;
    x2 = x2 / 6.0f;
    float scaled_hardsigmoid_x = inv_output_scale * x2 + output_zero_point;
    if (scaled_hardsigmoid_x < scaled_min) {
      scaled_hardsigmoid_x = scaled_min;
    }
    if (scaled_hardsigmoid_x > scaled_max) {
      scaled_hardsigmoid_x = scaled_max;
    }
    lookup_table[(uint32_t)i] = (uint8_t)lrintf(scaled_hardsigmoid_x);
  }

  hardsigmoid_op->channels = channels;

  hardsigmoid_op->ukernel_type = blacksmith_qnnp_ukernel_type_lut;
  hardsigmoid_op->format = blacksmith_qnnp_format_quint8;

  *hardsigmoid_out = hardsigmoid_op;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(hardsigmoid_op);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_hardsigmoid_nc_q8(
    blacksmith_qnnp_operator_t hardsigmoid,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_hardsigmoid_nc_q8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    hardsigmoid->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  hardsigmoid->batch_size = batch_size;
  hardsigmoid->input = input;
  hardsigmoid->input_pixel_stride = input_stride;
  hardsigmoid->output = output;
  hardsigmoid->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
