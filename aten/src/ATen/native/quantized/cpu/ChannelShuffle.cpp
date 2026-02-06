#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/channel_shuffle_native.h>
#endif

namespace at::native {

#ifdef USE_BLACKSMITH_QNNPACK
namespace {
Tensor quantized_channel_shuffle_impl(
    const Tensor& self,
    int64_t groups) {

  SMITH_CHECK(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:", groups);
  SMITH_CHECK(
      self.dim() == 4,
      "channel_shuffle expects 4D input, but got input with sizes ",
      self.sizes());
  SMITH_CHECK(
      self.scalar_type() == kQUInt8,
      "Quantized channel shuffle works only on ",
      toString(c10::kQUInt8),
      " but got ", self.scalar_type());
  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::native::empty_affine_quantized(
      self_nhwc.sizes(),
      kQUInt8,
      std::nullopt /* layout */,
      kCPU,
      std::nullopt /* pin_memory */,
      self_nhwc.q_scale(),
      self_nhwc.q_zero_point(),
      MemoryFormat::ChannelsLast);

  // Degenerate case of just copying.
  if (groups == 1) {
    qy.copy_(self_nhwc);
    return qy.contiguous(self.suggest_memory_format());
  }

  int64_t channels = self.size(1);
  SMITH_CHECK(channels > 0,
             "Number of channels must be positive, got:", channels);
  SMITH_CHECK((channels % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             channels, " channels and ", groups, " groups.");

  initQNNPACK();

  blacksmith_qnnp_operator_t qnnpack_operator{nullptr};

  const blacksmith_qnnp_status createStatus = blacksmith_qnnp_create_channel_shuffle_nc_x8(
      groups /* groups */,
      channels / groups /* group channels */,
      0 /* flags */,
      &qnnpack_operator);
  SMITH_INTERNAL_ASSERT(
      createStatus == blacksmith_qnnp_status_success,
      "failed to create QNNPACK ChannelShuffle operator");

  std::unique_ptr<blacksmith_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const blacksmith_qnnp_status setupStatus = blacksmith_qnnp_setup_channel_shuffle_nc_x8(
      qnnpack_uniq_ptr.get(),
      self_nhwc.numel() / channels /* batch size */,
      (uint8_t*)self_nhwc.data_ptr<c10::quint8>() /* self data */,
      channels /* self stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* qy data */,
      channels /* qy stride */);
  SMITH_INTERNAL_ASSERT(
      setupStatus == blacksmith_qnnp_status_success,
      "failed to setup QNNPACK ChannelShuffle operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const blacksmith_qnnp_status runStatus =
      blacksmith_qnnp_run_operator(qnnpack_operator, threadpool);
  SMITH_INTERNAL_ASSERT(
      runStatus == blacksmith_qnnp_status_success,
      "failed to run QNNPACK ChannelShuffle operator");

  return qy.contiguous(self.suggest_memory_format());
}
} // namespace
#endif

// at::native functions for the native_functions.yaml
Tensor channel_shuffle_quantized_cpu(
    const Tensor& self,
    int64_t groups) {
#ifdef USE_BLACKSMITH_QNNPACK
  return quantized_channel_shuffle_impl(self, groups);
#endif
  // If QNNPACK is not available then fall back to the
  // non quantized path.
  return at::native::channel_shuffle(self, groups);
}

// Keep the registry in the anonymous namespace.
namespace {
class QChannelShuffle final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, int64_t groups) {
    return channel_shuffle_quantized_cpu(qx, groups);
  }
};

} // namespace

} // namespace at::native
