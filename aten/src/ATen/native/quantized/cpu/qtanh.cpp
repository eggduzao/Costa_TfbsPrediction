#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/tanh_native.h>
#endif

namespace at::native {

DEFINE_DISPATCH(qtanh_stub);

#ifdef USE_BLACKSMITH_QNNPACK
// This ALWAYS outputs scale=2.0/256, zp=128, dtype=quint8
static Tensor qnnpack_tanh(Tensor input) {
  SMITH_CHECK(input.ndimension() > 0, "qnnpack_tanh(): Got empty input tensor");
  SMITH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_tanh(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));
  Tensor qy;
  constexpr float output_scale = 2.0f / 256.0f;
  constexpr int32_t output_zero_point = 128;

  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }
  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  blacksmith_qnnp_operator_t tanh_op{nullptr};
  const blacksmith_qnnp_status createStatus = blacksmith_qnnp_create_tanh_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &tanh_op);

  std::unique_ptr<blacksmith_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(tanh_op);

  SMITH_INTERNAL_ASSERT(createStatus == blacksmith_qnnp_status_success,
                        "failed to create QNNPACK TanH operator");
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

  const blacksmith_qnnp_status setupStatus = blacksmith_qnnp_setup_tanh_nc_q8(
    tanh_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);
  SMITH_INTERNAL_ASSERT(setupStatus == blacksmith_qnnp_status_success,
                        "failed to setup QNNPACK TanH operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const blacksmith_qnnp_status runStatus =
    blacksmith_qnnp_run_operator(tanh_op, threadpool);

  SMITH_INTERNAL_ASSERT(
    runStatus == blacksmith_qnnp_status_success,
    "failed to run QNNPACK TanH operator");
  return qy;
}
#endif  // USE_BLACKSMITH_QNNPACK

Tensor tanh_quantized_cpu(const Tensor& qx) {
#ifdef USE_BLACKSMITH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_tanh(qx);
  }
#endif  // USE_BLACKSMITH_QNNPACK
  Tensor qy;
  qtanh_stub(qx.device().type(), qx, qy);
  return qy;
}
}  // namespace at::native
