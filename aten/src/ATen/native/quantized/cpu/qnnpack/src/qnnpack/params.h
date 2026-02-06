/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>

#include <cpuinfo.h>

struct blacksmith_qnnp_fp16_clamping_params {
  uint16_t scale;
  uint16_t max;
  uint16_t min;
};

struct blacksmith_qnnp_fp32_clamping_params {
  float max;
  float min;
};

union blacksmith_qnnp_fp32_requantization_params {
  struct {
    float* scales;
    uint8_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
    float min_less_zero_point;
    float max_less_zero_point;
    float magic;
    int32_t magic_less_zero_point;
  } scalar;
  struct {
    float* scales;
    float max;
    float min;
    float magic;
    int32_t magic_less_zero_point;
  } neon;
  struct {
    float* scales;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neonv8;
  struct {
    BLACKSMITH_QNNP_ALIGN(16) float* scales;
    BLACKSMITH_QNNP_ALIGN(16) int16_t zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
  struct {
    BLACKSMITH_QNNP_ALIGN(16) float* scales;
    BLACKSMITH_QNNP_ALIGN(16) float min_less_zero_point[4];
    BLACKSMITH_QNNP_ALIGN(16) float max_less_zero_point[4];
    BLACKSMITH_QNNP_ALIGN(16) float magic[4];
    BLACKSMITH_QNNP_ALIGN(16) int32_t magic_less_zero_point[4];
  } psimd;
};

union blacksmith_qnnp_precise_requantization_params {
  struct {
    uint32_t multiplier;
    uint32_t rounding_lo;
    uint32_t rounding_hi;
    uint32_t shift_less_32;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neon;
  struct {
    BLACKSMITH_QNNP_ALIGN(16) uint32_t multiplier[4];
    BLACKSMITH_QNNP_ALIGN(16) uint64_t rounding[2];
    BLACKSMITH_QNNP_ALIGN(16) uint32_t shift[4];
    BLACKSMITH_QNNP_ALIGN(16) int16_t zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
};

union blacksmith_qnnp_q31_requantization_params {
  struct {
    int32_t multiplier;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    uint32_t shift;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t max;
    uint8_t min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    BLACKSMITH_QNNP_ALIGN(16) uint32_t multiplier[4];
    BLACKSMITH_QNNP_ALIGN(16) uint64_t rounding[2];
    BLACKSMITH_QNNP_ALIGN(16) int32_t remainder_mask[4];
    BLACKSMITH_QNNP_ALIGN(16) int32_t remainder_threshold[4];
    BLACKSMITH_QNNP_ALIGN(16) uint64_t shift[2];
    BLACKSMITH_QNNP_ALIGN(16) int16_t zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

union blacksmith_qnnp_conv_quantization_params {
  struct {
    const uint8_t* kernel_zero_points;
    int32_t input_zero_point;
    const float* requantization_scales;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    const uint8_t* kernel_zero_points;
    int16_t input_zero_point;
    const float* requantization_scales;
    int16_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
    // Following four are for nearest-ties-to-even
    // rounding in aarch32. This saves some instructions
    // needed otherwise.
    float vfmax;
    float vfmin;
    float vfmagic;
    int32_t vimagic;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    BLACKSMITH_QNNP_ALIGN(16) const uint8_t* kernel_zero_points;
    BLACKSMITH_QNNP_ALIGN(16) int16_t input_zero_point[8];
    const BLACKSMITH_QNNP_ALIGN(16) float* requantization_scales;
    BLACKSMITH_QNNP_ALIGN(16) int16_t output_zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

struct blacksmith_qnnp_conv_dynamic_quantization_params {
  int16_t input_zero_point;
  const uint8_t* kernel_zero_points;
  const float* multipliers;
};

union blacksmith_qnnp_requantization_params {
  union blacksmith_qnnp_precise_requantization_params precise;
  union blacksmith_qnnp_fp32_requantization_params fp32;
  union blacksmith_qnnp_q31_requantization_params q31;
};

union blacksmith_qnnp_add_quantization_params {
  struct {
    int32_t zero_point_product;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
    uint32_t shift;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    int32_t y_zero_point;
    int32_t y_max;
    int32_t y_min;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int16_t y_zero_point;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t right_shift;
    uint8_t y_max;
    uint8_t y_min;
  } neon;
#endif
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    BLACKSMITH_QNNP_ALIGN(16) int32_t zero_point_product[4];
    BLACKSMITH_QNNP_ALIGN(16) uint16_t a_multiplier_lo[8];
    BLACKSMITH_QNNP_ALIGN(16) uint16_t a_multiplier_hi[8];
    BLACKSMITH_QNNP_ALIGN(16) uint16_t b_multiplier_lo[8];
    BLACKSMITH_QNNP_ALIGN(16) uint16_t b_multiplier_hi[8];
    BLACKSMITH_QNNP_ALIGN(16) int32_t remainder_mask[4];
    BLACKSMITH_QNNP_ALIGN(16) int32_t remainder_threshold[4];
    BLACKSMITH_QNNP_ALIGN(16) int16_t y_zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t y_max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t y_min[16];
    uint32_t shift;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
  } sse2;
#endif
};

union blacksmith_qnnp_avgpool_quantization_params {
  struct {
    int32_t bias;
    float scale;
    int32_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int32_t bias;
    float scale;
    int16_t output_zero_point;
    uint8_t output_max;
    uint8_t output_min;
    // Following four are for nearest-ties-to-even
    // rounding in aarch32. This saves some instructions
    // needed otherwise.
    float vfmax;
    float vfmin;
    float vfmagic;
    int32_t vimagic;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    BLACKSMITH_QNNP_ALIGN(16) int32_t bias[4];
    BLACKSMITH_QNNP_ALIGN(16) float scale[4];
    BLACKSMITH_QNNP_ALIGN(16) int16_t output_zero_point[8];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif
};

union blacksmith_qnnp_u8_clamping_params {
  struct {
    int32_t output_max;
    int32_t output_min;
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t output_max;
    uint8_t output_min;
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_max[16];
    BLACKSMITH_QNNP_ALIGN(16) uint8_t output_min[16];
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

typedef void (*blacksmith_q8gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    size_t output_channel_index,
    const union blacksmith_qnnp_conv_quantization_params* quantization_params);

/*
  Q8 GEMM kernel with support for dynamic quantization.

  The w parameter designates weights, and is to be passed on to this kernel
  exactly as returned by the pack function.  The initial bias portion of
  this buffer will be ignored.

  The bias parameter, expects max(nr, 8) floating-point biases.  Technically
  the kernels only need nr biases from the buffer pointed to by this parameter,
  but end up reading at most 8 to keep the logic simple and fast.  Consequently,
  make sure this parameter has enough storage for 8 floating point numbers to
  avoid triggering out of bound errors.  The remaining 8 - nr biases, if any,
  will be unused.

  quantization_params contains the quantization parameters, namely input, and
  kernel zero points, and the multiplier.  The multiplier is expected to be
  equal to input_scale * kernel_scale.
*/

typedef void (*blacksmith_q8gemm_dq_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct blacksmith_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_dq_sparse_ukernel_function)(
    size_t mr,
    size_t nr,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* packed_w,
    const uint32_t* w_row_ptr,
    const uint32_t* w_block_ids_ptr,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct blacksmith_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_dq_sparse_packedA_w32_ukernel_function)(
    size_t mr,
    size_t nr,
    const uint8_t* a_packed,
    const uint8_t* packed_w,
    const uint32_t* w_row_ptr,
    const uint32_t* w_block_ids_ptr,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct blacksmith_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_dq_sparse_packedA_w16_ukernel_function)(
    size_t mr,
    size_t nr,
    const uint8_t* a_packed,
    const uint8_t* packed_w,
    const uint16_t* w_row_ptr,
    const uint16_t* w_block_ids_ptr,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct blacksmith_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_dq_sparse_packedA_w8_ukernel_function)(
    size_t mr,
    size_t nr,
    const uint8_t* a_packed,
    const uint8_t* packed_w,
    const uint8_t* w_row_ptr,
    const uint8_t* w_block_ids_ptr,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct blacksmith_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_sparse_packA_ukernel_function)(
    const size_t mr,
    const size_t K,
    const uint8_t* a,
    const size_t a_stride,
    uint8_t* a_packed);

typedef void (*blacksmith_q8conv_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** a,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    size_t output_channel_index,
    const union blacksmith_qnnp_conv_quantization_params* quantization_params);

typedef void (*blacksmith_q8gemm_xzp_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const int32_t* a_sum,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    const union blacksmith_qnnp_q31_requantization_params* requantization_params);

typedef void (*blacksmith_q8sum_rows_ukernel_function)(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    int32_t multiplier,
    int32_t* sums);

typedef void (*blacksmith_xzipc_ukernel_function)(size_t n, const void* x, void* y);

typedef void (
    *blacksmith_xzipv_ukernel_function)(size_t n, size_t m, const void* x, void* y);

typedef void (*blacksmith_x8lut_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint8_t* t,
    uint8_t* y);

typedef void (*blacksmith_sgemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t c_stride,
    const struct blacksmith_qnnp_fp32_clamping_params* clamping_params);

typedef void (*blacksmith_sconv_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t c_stride,
    const struct blacksmith_qnnp_fp32_clamping_params* clamping_params);

typedef void (*blacksmith_hgemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t c_stride,
    const struct blacksmith_qnnp_fp16_clamping_params* clamping_params);

typedef void (*blacksmith_q8dwconv2d_up_ukernel_function)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union blacksmith_qnnp_conv_quantization_params* quantization_params);

typedef void (*blacksmith_q8dwconv2d_mp_ukernel_function)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* buffer,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union blacksmith_qnnp_conv_quantization_params* quantization_params);

typedef void (*blacksmith_q8dwconv3d_mp_ukernel_function)(
    size_t channels,
    size_t output_height,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    int32_t* buffer,
    uint8_t* output,
    size_t input_row_stride,
    size_t input_col_stride,
    size_t output_increment,
    const union blacksmith_qnnp_conv_quantization_params* quantization_params);

typedef void (*blacksmith_q8gavgpool_up_ukernel_function)(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    uint8_t* y,
    const union blacksmith_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*blacksmith_q8gavgpool_mp_ukernel_function)(
    size_t m,
    size_t n,
    const uint8_t* x,
    size_t x_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    const union blacksmith_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*blacksmith_q8avgpool_up_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    const uint8_t* zero,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union blacksmith_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*blacksmith_q8avgpool_mp_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union blacksmith_qnnp_avgpool_quantization_params* quantization_params);

typedef void (*blacksmith_u8maxpool_ukernel_function)(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** x,
    uint8_t* y,
    size_t x_increment,
    size_t y_increment,
    const union blacksmith_qnnp_u8_clamping_params* params);

typedef void (*blacksmith_u8clamp_ukernel_function)(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union blacksmith_qnnp_u8_clamping_params* params);

typedef uint8_t (*blacksmith_u8rmax_ukernel_function)(size_t n, const uint8_t* x);

typedef void (*blacksmith_u8lut32norm_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y);

typedef void (*blacksmith_q8vadd_ukernel_function)(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union blacksmith_qnnp_add_quantization_params* quantization_params);

struct blacksmith_q8conv_parameters {
  blacksmith_q8gemm_ukernel_function gemm;
  blacksmith_q8conv_ukernel_function conv;
  blacksmith_q8gemm_dq_ukernel_function gemm_dq;
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
};

struct blacksmith_q8gemm_sparse_parameters {
  blacksmith_q8gemm_dq_sparse_ukernel_function gemm_dq;
  // w32, w16, and w8 refer to variants of the kernel which use uint32_t,
  // uint16_t, and uint8_t datatype for row values/col indices respectively
  blacksmith_q8gemm_dq_sparse_packedA_w32_ukernel_function packedA_w32_gemm_dq;
  blacksmith_q8gemm_dq_sparse_packedA_w16_ukernel_function packedA_w16_gemm_dq;
  blacksmith_q8gemm_dq_sparse_packedA_w8_ukernel_function packedA_w8_gemm_dq;
  blacksmith_q8gemm_sparse_packA_ukernel_function packA;
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
  uint8_t log2_mr;
  uint8_t log2_row_block_size;
  uint32_t row_block_size;
  uint32_t col_block_size;
};

struct blacksmith_q8conv_xzp_parameters {
  blacksmith_q8gemm_xzp_ukernel_function gemm;
  /* no conv ukernel */
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
  uint8_t kc;
  size_t kthreshold;
};

struct blacksmith_q8dwconv2d_up_parameters {
  blacksmith_q8dwconv2d_up_ukernel_function updw;
  blacksmith_q8dwconv2d_up_ukernel_function updw_per_channel;
  uint8_t cr;
};

struct blacksmith_q8dwconv2d_mp_parameters {
  blacksmith_q8dwconv2d_mp_ukernel_function mpdw;
  blacksmith_q8dwconv2d_mp_ukernel_function mpdw_per_channel;
  uint8_t cr;
};

struct blacksmith_q8dwconv3d_mp_parameters {
  blacksmith_q8dwconv3d_mp_ukernel_function mpdw;
  uint8_t cr;
};

struct blacksmith_q8sum_rows_parameters {
  blacksmith_q8sum_rows_ukernel_function sum_rows;
  uint32_t m;
};

struct blacksmith_q8gavgpool_parameters {
  blacksmith_q8gavgpool_up_ukernel_function ltnr;
  blacksmith_q8gavgpool_up_ukernel_function genr_lemr;
  blacksmith_q8gavgpool_mp_ukernel_function genr_gtmr;
  uint8_t mr;
  uint8_t nr;
};

struct blacksmith_q8avgpool_parameters {
  blacksmith_q8avgpool_up_ukernel_function ltkr;
  blacksmith_q8avgpool_up_ukernel_function gekr_lemr;
  blacksmith_q8avgpool_mp_ukernel_function gekr_gtmr;
  uint8_t mr;
  uint8_t qr;
  uint8_t kr;
};

struct blacksmith_u8maxpool_parameters {
  blacksmith_u8maxpool_ukernel_function ltkr;
  blacksmith_u8maxpool_ukernel_function gekr;
  uint8_t mr;
  uint8_t qr;
  uint8_t kr;
};

struct blacksmith_x8zip_parameters {
  blacksmith_xzipc_ukernel_function x2;
  blacksmith_xzipc_ukernel_function x3;
  blacksmith_xzipc_ukernel_function x4;
  blacksmith_xzipv_ukernel_function xm;
};

struct blacksmith_qnnp_parameters {
  struct blacksmith_q8conv_parameters q8conv;
  struct blacksmith_q8gemm_sparse_parameters q8gemm_sparse_c1x4;
  struct blacksmith_q8gemm_sparse_parameters q8gemm_sparse_c8x1;
  struct blacksmith_q8conv_xzp_parameters q8conv_xzp;
  struct blacksmith_q8dwconv2d_up_parameters q8dw9;
  struct blacksmith_q8dwconv2d_mp_parameters q8dw25;
  struct blacksmith_q8dwconv3d_mp_parameters q8dw27;
  struct blacksmith_q8sum_rows_parameters q8sum_rows;
  blacksmith_q8vadd_ukernel_function q8vadd;
  struct blacksmith_q8gavgpool_parameters q8gavgpool;
  struct blacksmith_q8avgpool_parameters q8avgpool;
  struct blacksmith_u8maxpool_parameters u8maxpool;
  blacksmith_u8lut32norm_ukernel_function u8lut32norm;
  blacksmith_u8clamp_ukernel_function u8clamp;
  blacksmith_u8rmax_ukernel_function u8rmax;
  struct blacksmith_x8zip_parameters x8zip;
  blacksmith_x8lut_ukernel_function x8lut;
  bool initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

extern struct blacksmith_qnnp_parameters blacksmith_qnnp_params;

#ifdef __cplusplus
}
#endif
