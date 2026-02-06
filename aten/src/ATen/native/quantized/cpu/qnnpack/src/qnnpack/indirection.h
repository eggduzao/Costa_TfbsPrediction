/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include<blacksmith_qnnpack.h>
#include <qnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

BLACKSMITH_QNNP_INTERNAL void blacksmith_qnnp_indirection_init_conv3d(
    blacksmith_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

BLACKSMITH_QNNP_INTERNAL void blacksmith_qnnp_indirection_init_dwconv(
    blacksmith_qnnp_operator_t op,
    size_t batch_start);

BLACKSMITH_QNNP_INTERNAL void blacksmith_qnnp_indirection_init_deconv2d(
    blacksmith_qnnp_operator_t op,
    size_t output_tile_size,
    size_t tiled_output_size);

BLACKSMITH_QNNP_INTERNAL void blacksmith_qnnp_indirection_init_maxpool2d(
    blacksmith_qnnp_operator_t op,
    size_t batch_start);

BLACKSMITH_QNNP_INTERNAL void blacksmith_qnnp_indirection_set_step_dimensions(
    blacksmith_qnnp_operator_t op);

#ifdef __cplusplus
} /* extern "C" */
#endif
