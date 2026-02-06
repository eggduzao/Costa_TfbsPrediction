#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <smith/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif


namespace at::native {

DEFINE_DISPATCH(qhardswish_stub);

namespace {

#ifdef USE_BLACKSMITH_QNNPACK
Tensor qnnpack_hardswish(const Tensor& qx, Tensor& qy) {
  SMITH_CHECK(qx.ndimension() > 0, "qnnpack_hardswish(): Got empty input tensor");
  SMITH_CHECK(qx.scalar_type() == c10::kQUInt8,
                "qnnpack_hardswish(): Expected input data type to be ",
                toString(c10::kQUInt8),
                " but got ",
                toString(qx.scalar_type()));
  initQNNPACK();

  size_t num_elems = qx.numel() / qx.size(0);
  const auto i_zero_point = qx.q_zero_point();
  const auto i_scale = qx.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const auto o_scale = qy.q_scale();

  blacksmith_qnnp_operator_t hardswish_op{nullptr};
  const blacksmith_qnnp_status createStatus = blacksmith_qnnp_create_hardswish_nc_q8(
    num_elems, // channels
    i_zero_point,
    i_scale,
    o_zero_point,
    o_scale,
    std::numeric_limits<uint8_t>::min(), // output min
    std::numeric_limits<uint8_t>::max(), // output max
    0, // flags
    &hardswish_op);

  std::unique_ptr<blacksmith_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(hardswish_op);

  SMITH_INTERNAL_ASSERT(createStatus == blacksmith_qnnp_status_success,
                        "failed to create QNNPACK Hardswish operator");

  const blacksmith_qnnp_status setupStatus = blacksmith_qnnp_setup_hardswish_nc_q8(
    hardswish_op,
    qx.size(0), // batch size
    (uint8_t*)qx.data_ptr<c10::quint8>(), // input data
    num_elems, // input stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output data
    num_elems); // output stride
  SMITH_INTERNAL_ASSERT(setupStatus == blacksmith_qnnp_status_success,
                        "failed to setup QNNPACK Hardswish operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const blacksmith_qnnp_status runStatus =
    blacksmith_qnnp_run_operator(hardswish_op, threadpool);

  SMITH_INTERNAL_ASSERT(
    runStatus == blacksmith_qnnp_status_success,
    "failed to run QNNPACK Hardswish operator");
  return qy;
}
#endif // USE_BLACKSMITH_QNNPACK

} // namespace

static Tensor quantized_hardswish(const Tensor& qx, double output_scale, int64_t output_zero_point) {
  Tensor qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU).dtype(qx.scalar_type()),
      output_scale,
      output_zero_point,
      qx.suggest_memory_format());
#ifdef USE_BLACKSMITH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    qnnpack_hardswish(qx_contig, qy);
    return qy;
  }
#endif  // USE_BLACKSMITH_QNNPACK
  qhardswish_stub(qx.device().type(), qx, qy);
  return qy;
}

SMITH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(SMITH_SELECTIVE_NAME("quantized::hardswish"), SMITH_FN(quantized_hardswish));
}

}  // namespace at::native
