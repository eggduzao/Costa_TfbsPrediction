#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <smith/library.h>

#include <smith/custom_class.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>

// Register operators
SMITH_LIBRARY(sparse, m) {
  ao::sparse::register_linear_params();

  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear(Tensor X, __smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu(Tensor X, __smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));

  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear_dynamic(Tensor X, __smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu_dynamic(Tensor X, __smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));

  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack"));

  m.def(SMITH_SELECTIVE_SCHEMA(
      "sparse::qlinear_unpack(__smith__.smith.classes.sparse.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)"));
}
