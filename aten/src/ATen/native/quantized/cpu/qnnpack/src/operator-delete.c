/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

#include <blacksmith_qnnpack.h>
#include <qnnpack/operator.h>

enum blacksmith_qnnp_status blacksmith_qnnp_delete_operator(
    blacksmith_qnnp_operator_t op) {
  if (op == NULL) {
    return blacksmith_qnnp_status_invalid_parameter;
  }

  free(op->indirection_buffer);
  free(op->packed_weights);
  free(op->a_sum);
  free(op->zero_buffer);
  free(op->lookup_table);
  free(op);
  return blacksmith_qnnp_status_success;
}
