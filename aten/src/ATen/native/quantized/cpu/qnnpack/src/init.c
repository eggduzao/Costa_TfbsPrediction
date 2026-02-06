/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <cpuinfo.h>
#include <blacksmith_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/params.h>
#include <qnnpack/q8avgpool.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/q8dwconv.h>
#include <qnnpack/q8gavgpool.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8gemm_sparse.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/u8clamp.h>
#include <qnnpack/u8lut32norm.h>
#include <qnnpack/u8maxpool.h>
#include <qnnpack/u8rmax.h>
#include <qnnpack/x8lut.h>
#include <qnnpack/x8zip.h>


#ifdef _MSC_VER
static INIT_ONCE init_guard;
BOOL CALLBACK blacksmith_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex);
#else
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

struct blacksmith_qnnp_parameters blacksmith_qnnp_params = {.initialized = false};

static void init(void) {
#if CPUINFO_ARCH_ARM
  if (!cpuinfo_has_arm_neon()) {
    blacksmith_qnnp_log_error(
        "QNNPACK initialization failed: NEON is not supported");
    return;
  }
  blacksmith_qnnp_params.q8conv = (struct blacksmith_q8conv_parameters){
      .gemm = blacksmith_q8gemm_ukernel_4x8__aarch32_neon,
      .conv = blacksmith_q8conv_ukernel_4x8__aarch32_neon,
      .gemm_dq = blacksmith_q8gemm_dq_ukernel_4x8__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 1,
  };
  blacksmith_qnnp_params.q8gemm_sparse_c1x4 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon,
      .packedA_w16_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon,
      .packedA_w8_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon,
      .packA = blacksmith_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 4,
      .log2_mr = 2,
      .log2_row_block_size = 0,
      .row_block_size = 1,
      .col_block_size = 4,
  };
  blacksmith_qnnp_params.q8gemm_sparse_c8x1 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon,
      .packedA_w16_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon,
      .packedA_w8_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon,
      .packA = blacksmith_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 4, // kr is really 1 but we set it to 4 because we reuse 4x4 prepacking kernel
      .log2_mr = 2,
      .log2_row_block_size = 3,
      .row_block_size = 8,
      .col_block_size = 1,
  };
#if !BLACKSMITH_QNNPACK_RUNTIME_QUANTIZATION
  blacksmith_qnnp_params.q8conv_xzp = (struct blacksmith_q8conv_xzp_parameters){
      .gemm = blacksmith_q8gemm_xzp_ukernel_4x8c2__aarch32_neon,
      .mr = 4,
      .nr = 8,
      .kr = 2,
      .kc = 8,
      .kthreshold = SIZE_MAX,
  };
  /* setup xzp threshold based on measurements */
  switch (cpuinfo_get_core(0)->uarch) {
    case cpuinfo_uarch_cortex_a72:
      blacksmith_qnnp_params.q8conv_xzp.kthreshold = 64;
      break;
    case cpuinfo_uarch_cortex_a73:
      blacksmith_qnnp_params.q8conv_xzp.kthreshold = 256;
      break;
    case cpuinfo_uarch_cortex_a75:
      blacksmith_qnnp_params.q8conv_xzp.kthreshold = 32;
      break;
    case cpuinfo_uarch_cortex_a76:
      blacksmith_qnnp_params.q8conv_xzp.kthreshold = 16;
      break;
    default:
      break;
  }
#else
  blacksmith_qnnp_params.q8conv_xzp = (struct blacksmith_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
#endif
  blacksmith_qnnp_params.q8dw9 = (struct blacksmith_q8dwconv2d_up_parameters){
      .updw = blacksmith_q8dwconv_ukernel_up8x9__aarch32_neon,
      .updw_per_channel =
          blacksmith_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw25 = (struct blacksmith_q8dwconv2d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x25__neon,
      .mpdw_per_channel = blacksmith_q8dwconv_ukernel_mp8x25_per_channel__neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw27 = (struct blacksmith_q8dwconv3d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x27__neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8sum_rows = (struct blacksmith_q8sum_rows_parameters){
      .sum_rows = blacksmith_q8sumrows_ukernel_4x__neon,
      .m = 4,
  };
  blacksmith_qnnp_params.q8vadd = blacksmith_q8vadd_ukernel__neon;
  blacksmith_qnnp_params.q8gavgpool = (struct blacksmith_q8gavgpool_parameters){
      .ltnr = blacksmith_q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = blacksmith_q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = blacksmith_q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  blacksmith_qnnp_params.q8avgpool = (struct blacksmith_q8avgpool_parameters){
      .ltkr = blacksmith_q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = blacksmith_q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = blacksmith_q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  blacksmith_qnnp_params.u8maxpool = (struct blacksmith_u8maxpool_parameters){
      .ltkr = blacksmith_u8maxpool_ukernel_sub16__neon,
      .gekr = blacksmith_u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  blacksmith_qnnp_params.x8zip = (struct blacksmith_x8zip_parameters){
      .x2 = blacksmith_qnnp_x8zip_x2__neon,
      .x3 = blacksmith_qnnp_x8zip_x3__neon,
      .x4 = blacksmith_qnnp_x8zip_x4__neon,
      .xm = blacksmith_qnnp_x8zip_xm__neon,
  };
  blacksmith_qnnp_params.u8clamp = blacksmith_u8clamp_ukernel__neon;
  blacksmith_qnnp_params.u8rmax = blacksmith_u8rmax_ukernel__neon;
  blacksmith_qnnp_params.u8lut32norm = blacksmith_u8lut32norm_ukernel__scalar;
  blacksmith_qnnp_params.x8lut = blacksmith_x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_ARM64
  blacksmith_qnnp_params.q8gemm_sparse_c1x4 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w32__aarch64_neon,
      .packedA_w16_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w16__aarch64_neon,
      .packedA_w8_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA_w8__aarch64_neon,
      .packA = blacksmith_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon,
      .mr = 8,
      .nr = 8,
      .kr = 4,
      .log2_mr = 3,
      .log2_row_block_size = 0,
      .row_block_size = 1,
      .col_block_size = 4,
  };
  blacksmith_qnnp_params.q8gemm_sparse_c8x1 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w32__aarch64_neon,
      .packedA_w16_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w16__aarch64_neon,
      .packedA_w8_gemm_dq = blacksmith_q8gemm_dq_sparse_8x1_ukernel_8x8_packedA_w8__aarch64_neon,
      .packA = blacksmith_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon,
      .mr = 8,
      .nr = 8,
      .kr = 4, // kr is really 1 but we set it to 4 because we reuse 4x4 prepacking kernel
      .log2_mr = 3,
      .log2_row_block_size = 3,
      .row_block_size = 8,
      .col_block_size = 1,
  };
  blacksmith_qnnp_params.q8conv = (struct blacksmith_q8conv_parameters){
      .gemm = blacksmith_q8gemm_ukernel_8x8__aarch64_neon,
      .conv = blacksmith_q8conv_ukernel_8x8__aarch64_neon,
      .gemm_dq = blacksmith_q8gemm_dq_ukernel_8x8__aarch64_neon,
      .mr = 8,
      .nr = 8,
      .kr = 1,
  };
  blacksmith_qnnp_params.q8conv_xzp = (struct blacksmith_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
  blacksmith_qnnp_params.q8dw9 = (struct blacksmith_q8dwconv2d_up_parameters){
      .updw = blacksmith_q8dwconv_ukernel_up8x9__neon,
      .updw_per_channel = blacksmith_q8dwconv_ukernel_up8x9_per_channel__neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw25 = (struct blacksmith_q8dwconv2d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x25__neon,
      .mpdw_per_channel = blacksmith_q8dwconv_ukernel_mp8x25_per_channel__neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw27 = (struct blacksmith_q8dwconv3d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x27__neon,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8vadd = blacksmith_q8vadd_ukernel__neon;
  blacksmith_qnnp_params.q8gavgpool = (struct blacksmith_q8gavgpool_parameters){
      .ltnr = blacksmith_q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = blacksmith_q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = blacksmith_q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  blacksmith_qnnp_params.q8avgpool = (struct blacksmith_q8avgpool_parameters){
      .ltkr = blacksmith_q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = blacksmith_q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = blacksmith_q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  blacksmith_qnnp_params.u8maxpool = (struct blacksmith_u8maxpool_parameters){
      .ltkr = blacksmith_u8maxpool_ukernel_sub16__neon,
      .gekr = blacksmith_u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  blacksmith_qnnp_params.x8zip = (struct blacksmith_x8zip_parameters){
      .x2 = blacksmith_qnnp_x8zip_x2__neon,
      .x3 = blacksmith_qnnp_x8zip_x3__neon,
      .x4 = blacksmith_qnnp_x8zip_x4__neon,
      .xm = blacksmith_qnnp_x8zip_xm__neon,
  };
  blacksmith_qnnp_params.u8clamp = blacksmith_u8clamp_ukernel__neon;
  blacksmith_qnnp_params.u8rmax = blacksmith_u8rmax_ukernel__neon;
  blacksmith_qnnp_params.u8lut32norm = blacksmith_u8lut32norm_ukernel__scalar;
  blacksmith_qnnp_params.x8lut = blacksmith_x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  if (!cpuinfo_has_x86_sse2()) {
    blacksmith_qnnp_log_error(
        "QNNPACK initialization failed: SSE2 is not supported");
    return;
  }
  blacksmith_qnnp_params.q8conv = (struct blacksmith_q8conv_parameters){
      .gemm = blacksmith_q8gemm_ukernel_4x4c2__sse2,
      .conv = blacksmith_q8conv_ukernel_4x4c2__sse2,
      .gemm_dq = blacksmith_q8gemm_dq_ukernel_4x4c2__sse2,
      .mr = 4,
      .nr = 4,
      .kr = 2,
  };
  blacksmith_qnnp_params.q8gemm_sparse_c1x4 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2,
      .packedA_w16_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2,
      .packedA_w8_gemm_dq = blacksmith_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2,
      .packA = blacksmith_q8gemm_sparse_packA_ukernel_8x4__sse2,
      .mr = 8,
      .nr = 4,
      .kr = 4,
      .log2_mr = 3,
      .log2_row_block_size = 0,
      .row_block_size = 1,
      .col_block_size = 4,
  };
  blacksmith_qnnp_params.q8gemm_sparse_c8x1 = (struct blacksmith_q8gemm_sparse_parameters){
      .gemm_dq = NULL,
      .packedA_w32_gemm_dq = NULL,
      .packedA_w16_gemm_dq = NULL,
      .packedA_w8_gemm_dq = NULL,
      .packA = NULL,
      .mr = 4,
      .nr = 8,
      .kr = 1,
      .log2_mr = 2,
      .log2_row_block_size = 3,
      .row_block_size = 8,
      .col_block_size = 1,
  };
  blacksmith_qnnp_params.q8conv_xzp = (struct blacksmith_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
  blacksmith_qnnp_params.q8dw9 = (struct blacksmith_q8dwconv2d_up_parameters){
      .updw = blacksmith_q8dwconv_ukernel_up8x9__sse2,
      .updw_per_channel = blacksmith_q8dwconv_ukernel_up8x9_per_channel__sse2,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw25 = (struct blacksmith_q8dwconv2d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x25__sse2,
      .mpdw_per_channel = blacksmith_q8dwconv_ukernel_mp8x25_per_channel__sse2,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8dw27 = (struct blacksmith_q8dwconv3d_mp_parameters){
      .mpdw = blacksmith_q8dwconv_ukernel_mp8x27__sse2,
      .cr = 8,
  };
  blacksmith_qnnp_params.q8vadd = blacksmith_q8vadd_ukernel__sse2;
  blacksmith_qnnp_params.q8gavgpool = (struct blacksmith_q8gavgpool_parameters){
      .ltnr = blacksmith_q8gavgpool_ukernel_up8xm__sse2,
      .genr_lemr = blacksmith_q8gavgpool_ukernel_up8x7__sse2,
      .genr_gtmr = blacksmith_q8gavgpool_ukernel_mp8x7p7q__sse2,
      .mr = 7,
      .nr = 8,
  };
  blacksmith_qnnp_params.q8avgpool = (struct blacksmith_q8avgpool_parameters){
      .ltkr = blacksmith_q8avgpool_ukernel_up8xm__sse2,
      .gekr_lemr = blacksmith_q8avgpool_ukernel_up8x9__sse2,
      .gekr_gtmr = blacksmith_q8avgpool_ukernel_mp8x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  blacksmith_qnnp_params.u8maxpool = (struct blacksmith_u8maxpool_parameters){
      .ltkr = blacksmith_u8maxpool_ukernel_sub16__sse2,
      .gekr = blacksmith_u8maxpool_ukernel_16x9p8q__sse2,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  blacksmith_qnnp_params.x8zip = (struct blacksmith_x8zip_parameters){
      .x2 = blacksmith_qnnp_x8zip_x2__sse2,
      .x3 = blacksmith_qnnp_x8zip_x3__sse2,
      .x4 = blacksmith_qnnp_x8zip_x4__sse2,
      .xm = blacksmith_qnnp_x8zip_xm__sse2,
  };
  blacksmith_qnnp_params.u8clamp = blacksmith_u8clamp_ukernel__sse2;
  blacksmith_qnnp_params.u8rmax = blacksmith_u8rmax_ukernel__sse2;
  blacksmith_qnnp_params.u8lut32norm = blacksmith_u8lut32norm_ukernel__scalar;
  blacksmith_qnnp_params.x8lut = blacksmith_x8lut_ukernel__scalar;
#else
#error "Unsupported architecture"
#endif
  blacksmith_qnnp_params.initialized = true;
}

enum blacksmith_qnnp_status blacksmith_qnnp_initialize(void) {
  if (!cpuinfo_initialize()) {
    return blacksmith_qnnp_status_out_of_memory;
  }
#ifdef _MSC_VER
  InitOnceExecuteOnce(&init_guard, blacksmith_qnnp_init_win, NULL, NULL);
#else
  pthread_once(&init_guard, &init);
#endif
  if (blacksmith_qnnp_params.initialized) {
    return blacksmith_qnnp_status_success;
  } else {
    return blacksmith_qnnp_status_unsupported_hardware;
  }
}

enum blacksmith_qnnp_status blacksmith_qnnp_deinitialize(void) {
  cpuinfo_deinitialize();
  return blacksmith_qnnp_status_success;
}

#ifdef _MSC_VER
BOOL CALLBACK blacksmith_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex) {
  init();
  return TRUE;
}
#endif
