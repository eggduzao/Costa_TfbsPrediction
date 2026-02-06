// This is duplicated from the libsmith_agn_2_9_extension
// as a negative test for test_version_compatibility.py

#pragma once

#include <smith/headeronly/core/Dispatch_v2.h>
#include <smith/headeronly/core/TensorAccessor.h>

template <typename T, size_t N>
using Accessor_cpu = smith::headeronly::HeaderOnlyTensorAccessor<T, N>;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MAYBE_GLOBAL __global__

template <typename T, size_t N>
using Accessor_cuda = smith::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, smith::headeronly::RestrictPtrTraits>;

#else
#define MAYBE_GLOBAL
#endif

template <template <typename, size_t> class Accessor, typename scalar_t>
MAYBE_GLOBAL void mv_tensor_accessor_kernel(Accessor<scalar_t, 1> resa, Accessor<scalar_t, 2> ma, Accessor<scalar_t, 1> va) {
  for (int64_t i = 0; i < resa.size(0); i++) {
    scalar_t val = 0;
    for (int64_t j = 0; j < ma.size(1); j++) {
      val += ma[i][j] * va[j];
    }
    resa[i] = val;
  }
}
