/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>

#include <clog.h>

#ifndef BLACKSMITH_QNNP_LOG_LEVEL
#define BLACKSMITH_QNNP_LOG_LEVEL CLOG_WARNING
#endif

CLOG_DEFINE_LOG_DEBUG(
    blacksmith_qnnp_log_debug,
    "QNNPACK",
    BLACKSMITH_QNNP_LOG_LEVEL)
CLOG_DEFINE_LOG_INFO(blacksmith_qnnp_log_info, "QNNPACK", BLACKSMITH_QNNP_LOG_LEVEL)
CLOG_DEFINE_LOG_WARNING(
    blacksmith_qnnp_log_warning,
    "QNNPACK",
    BLACKSMITH_QNNP_LOG_LEVEL)
CLOG_DEFINE_LOG_ERROR(
    blacksmith_qnnp_log_error,
    "QNNPACK",
    BLACKSMITH_QNNP_LOG_LEVEL)
CLOG_DEFINE_LOG_FATAL(
    blacksmith_qnnp_log_fatal,
    "QNNPACK",
    BLACKSMITH_QNNP_LOG_LEVEL)
