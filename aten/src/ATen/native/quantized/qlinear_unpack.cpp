/*
The dispatch registrations at the end of this file applies to fbgemm, qnnpack, and cudnn backends.
The correct unpack backend function is determined using runtime polymorphism through the packed_weight pointer,
which is of type intrusive_ptr<LinearPackedParamsBase> and points to either a PackedLinearWeightsQnnp,
PackedLinearWeights (Fbgemm), or PackedLinearWeightsCudnn at runtime, which all inherit from LinearPackedParamsBase.
The implementations for the unpack functions can be found in /cpu/LinearUnpackImpl.cpp, for fbgemm&qnnpack
and /cudnn/linear_unpack_impl.cpp, for cudnn.
*/
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/library.h>
#include <ATen/native/quantized/PackedParams.h>
#include <smith/custom_class.h>
#include <smith/library.h>

namespace at::native {
namespace {

class QLinearUnpackWeightInt8 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightFp16 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    auto& ctx = at::globalContext();

    SMITH_CHECK(
        ctx.qEngine() != at::QEngine::QNNPACK,
        "quantized::linear_unpack_fp16 is currently "
        "not supported by QNNPACK");

    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightInt8Legacy final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    SMITH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

class QLinearUnpackWeightFp16Legacy final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    SMITH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

SMITH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(SMITH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), SMITH_FN(QLinearUnpackWeightInt8Legacy::run));
  m.impl(SMITH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), SMITH_FN(QLinearUnpackWeightFp16Legacy::run));
}

SMITH_LIBRARY_IMPL(quantized, CatchAll, m) {
  register_linear_params();
  m.impl(SMITH_SELECTIVE_NAME("quantized::linear_unpack"), SMITH_FN(QLinearUnpackWeightInt8::run));
  m.impl(SMITH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), SMITH_FN(QLinearUnpackWeightFp16::run));
}

} // namespace
} // namespace at::native
