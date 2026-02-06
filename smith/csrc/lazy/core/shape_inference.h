#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <smith/csrc/lazy/backend/backend_data.h>
#include <smith/csrc/lazy/core/ir.h>
#include <smith/csrc/lazy/core/shape.h>
#include <smith/csrc/lazy/core/tensor.h>
#include <optional>
#include <vector>

namespace smith::lazy {
// Turn clang-format off, as we rely on the whole signature being on one line
// for codegen.
// clang-format off
SMITH_API std::vector<smith::lazy::Shape> compute_shape__adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
SMITH_API std::vector<smith::lazy::Shape> compute_shape__adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape__adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size);
SMITH_API std::vector<smith::lazy::Shape> compute_shape__adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_abs(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, double p, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_cat(at::TensorList tensors, int64_t dim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_cholesky(const at::Tensor & self, bool upper);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_clamp_min(const at::Tensor & self, const at::Scalar & min);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_clone(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_expand(const at::Tensor & self, at::IntArrayRef size, bool implicit);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_expand(const at::Tensor & self, c10::SymIntArrayRef size, bool implicit);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_flip(const at::Tensor & self, at::IntArrayRef dims);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_glu_jvp(const at::Tensor & glu, const at::Tensor & x, const at::Tensor & dx, int64_t dim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_inverse(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_isnan(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_log_sigmoid_forward(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_logdet(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_logical_and(const at::Tensor & self, const at::Tensor & other);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_logical_not(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_logical_or(const at::Tensor & self, const at::Tensor & other);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_logical_xor(const at::Tensor & self, const at::Tensor & other);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_max(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_mean(const at::Tensor & self, ::std::optional<at::ScalarType> dtype);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_min(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_mv(const at::Tensor & self, const at::Tensor & vec);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_batch_norm(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_dropout(const at::Tensor & input, double p, ::std::optional<bool> train);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_new_empty_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_nll_loss2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_nll_loss2d_forward(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_nonzero(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_normal_functional(const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_random(const at::Tensor & self, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_random(const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_relu(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_slogdet(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_smooth_l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_sort(const at::Tensor & self, int64_t dim, bool descending);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_stack(at::TensorList tensors, int64_t dim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_std(const at::Tensor & self, bool unbiased);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_std(const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_sum(const at::Tensor & self, ::std::optional<at::ScalarType> dtype);
SMITH_API std::vector<smith::lazy::Shape> compute_shape__to_copy(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_take(const at::Tensor & self, const at::Tensor & index);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_trace(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_zero(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_narrow_copy_symint(const at::Tensor & self, int64_t dim, int64_t start, c10::SymInt length);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_hardswish(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_hardswish_backward(const at::Tensor & grad_output, const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_selu(const at::Tensor & self);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_uniform(const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator);

// Non-Native ops
SMITH_API std::vector<Shape> compute_shape_scalar(const at::Scalar& value, const at::ScalarType& type);
SMITH_API std::vector<Shape> compute_shape_expand(const Output& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand);
SMITH_API std::vector<Shape> compute_shape_view(const Output& input0, const std::vector<int64_t>& output_sizes);
SMITH_API std::vector<Shape> compute_shape_cast(const Output& input0, const at::ScalarType& dtype, const ::std::optional<at::ScalarType>& stype);

// View Ops
// (Now that functionalization pass is used, we should kill these in a later PR)
SMITH_API std::vector<Shape> compute_shape_as_strided_view_update(const Output& target, const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);
SMITH_API std::vector<Shape> compute_shape_as_strided(const Output& input, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, const int64_t& storage_offset);
SMITH_API std::vector<Shape> compute_shape_diagonal_view_update(const Output& target, const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);
SMITH_API std::vector<Shape> compute_shape_diagonal(const Output& input, const int64_t& offset, const int64_t& dim1, const int64_t& dim2);
SMITH_API std::vector<Shape> compute_shape_narrow_view_update(const Output& input, const Output& source, const std::vector<int64_t>& base_indices);
SMITH_API std::vector<Shape> compute_shape_narrow(const Output& input, const std::vector<int64_t>& base_indices, const std::vector<int64_t>& sizes);
SMITH_API std::vector<Shape> compute_shape_permute(const Output& input, const std::vector<int64_t>& dims);
SMITH_API std::vector<Shape> compute_shape_resize(const Output& input, const std::vector<int64_t>& size);
SMITH_API std::vector<Shape> compute_shape_select_view_update(const Output& target, const Output& source, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);
SMITH_API std::vector<Shape> compute_shape_select(const Output& input, const int64_t& dim, const int64_t& start, const int64_t& end, const int64_t& stride);
SMITH_API std::vector<Shape> compute_shape_squeeze(const Output& input, const int& dim);
SMITH_API std::vector<Shape> compute_shape_unsqueeze(const Output& input, const int& dim);

SMITH_API std::vector<smith::lazy::Shape> compute_shape_select_scatter(const at::Tensor & self, const at::Tensor & src, int64_t dim, int64_t index);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_diagonal_scatter(const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_slice_scatter_symint(const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step);
SMITH_API std::vector<smith::lazy::Shape> compute_shape_as_strided_scatter_symint(const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset);
// clang-format on
} // namespace smith::lazy
