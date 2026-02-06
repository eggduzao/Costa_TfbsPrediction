#pragma once

#include <smith/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <smith/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <smith/csrc/inductor/aoti_runtime/thread_local.h>
#include <smith/csrc/inductor/aoti_smith/utils.h>

namespace smith::aot_inductor {
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_dim(handle, &dim));
  int64_t numel;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_SMITH_ERROR_CODE_CHECK(aoti_smith_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_SMITH_ERROR_CODE_CHECK(
      aoti_smith_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, uint64_t numel) {
  SMITH_CHECK(
      tensor.numel() == numel,
      "incorrect numel for input tensor. expected ",
      numel,
      ", got ",
      tensor.numel());
}
} // namespace smith::aot_inductor
