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

enum blacksmith_qnnp_status blacksmith_qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    blacksmith_qnnp_operator_t* clamp_out) {
  blacksmith_qnnp_operator_t clamp_op = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_invalid_parameter;

  if (channels == 0) {
    blacksmith_qnnp_log_error(
        "failed to create Clamp operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (output_min > output_max) {
    blacksmith_qnnp_log_error(
        "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8
        "] output range: range min must be below range max",
        output_min,
        output_max);
    goto error;
  }

  status = blacksmith_qnnp_status_out_of_memory;

  clamp_op = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (clamp_op == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  clamp_op->channels = channels;
  clamp_op->u8_clamping_params =
      blacksmith_qnnp_compute_u8_clamping_params(output_min, output_max);

  clamp_op->ukernel_type = blacksmith_qnnp_ukernel_type_clamp;
  clamp_op->format = blacksmith_qnnp_format_quint8;

  *clamp_out = clamp_op;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(clamp_op);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_clamp_nc_u8(
    blacksmith_qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    clamp->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  clamp->batch_size = batch_size;
  clamp->input = input;
  clamp->input_pixel_stride = input_stride;
  clamp->output = output;
  clamp->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
