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

enum blacksmith_qnnp_status blacksmith_qnnp_create_softargmax_nc_q8(
    size_t channels,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    blacksmith_qnnp_operator_t* softargmax_out) {
  blacksmith_qnnp_operator_t softargmax_op = NULL;
  enum blacksmith_qnnp_status status = blacksmith_qnnp_status_uninitialized;

  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_create_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = blacksmith_qnnp_status_invalid_parameter;

  if (channels == 0) {
    blacksmith_qnnp_log_error(
        "failed to create Soft ArgMax operator with %zu channels: number of channels must be non-zero",
        channels);
    goto error;
  }

  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g input scale: scale must be finite and positive",
        input_scale);
    goto error;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    blacksmith_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: scale must be finite and positive",
        output_scale);
    goto error;
  }

  status = blacksmith_qnnp_status_unsupported_parameter;

  if (output_scale != 0x1.0p-8f) {
    blacksmith_qnnp_log_error(
        "failed to create Soft ArgMax operator with %.7g output scale: only output scale of 1/256 is supported",
        output_scale);
    goto error;
  }

  if (output_zero_point != 0) {
    blacksmith_qnnp_log_error(
        "failed to create Soft ArgMax operator with %" PRIu8
        " output zero point: only output zero point of 0 is supported",
        output_zero_point);
    goto error;
  }

  status = blacksmith_qnnp_status_out_of_memory;

  softargmax_op = calloc(1, sizeof(struct blacksmith_qnnp_operator));
  if (softargmax_op == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate %zu bytes for blacksmith_qnnp_operator structure",
        sizeof(struct blacksmith_qnnp_operator));
    goto error;
  }

  softargmax_op->lookup_table = malloc(256 * sizeof(uint32_t));
  if (softargmax_op->lookup_table == NULL) {
    blacksmith_qnnp_log_error(
        "failed to allocate 256 bytes for Soft ArgMax lookup table");
    goto error;
  }

  uint32_t* lookup_table = softargmax_op->lookup_table;
  const double qscale =
      fmin(((double)UINT32_MAX) / (double)channels, 8388607.0);
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi =
        qscale * exp((double)(i - 255) * (double)input_scale);
    lookup_table[(uint32_t)i] = (uint32_t)lrint(scaled_exp_xi);
  }

  softargmax_op->channels = channels;

  softargmax_op->ukernel_type = blacksmith_qnnp_ukernel_type_softargmax;
  softargmax_op->format = blacksmith_qnnp_format_quint8;

  *softargmax_out = softargmax_op;
  return blacksmith_qnnp_status_success;

error:
  blacksmith_qnnp_delete_operator(softargmax_op);
  return status;
}

enum blacksmith_qnnp_status blacksmith_qnnp_setup_softargmax_nc_q8(
    blacksmith_qnnp_operator_t softargmax,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride) {
  if (!blacksmith_qnnp_params.initialized) {
    blacksmith_qnnp_log_error(
        "blacksmith_qnnp_setup_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
    return blacksmith_qnnp_status_uninitialized;
  }

  if (batch_size == 0) {
    softargmax->batch_size = 0;
    return blacksmith_qnnp_status_success;
  }

  softargmax->batch_size = batch_size;
  softargmax->input = input;
  softargmax->input_pixel_stride = input_stride;
  softargmax->output = output;
  softargmax->output_pixel_stride = output_stride;

  return blacksmith_qnnp_status_success;
}
