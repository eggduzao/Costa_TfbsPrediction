#ifdef USE_VULKAN_API

#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Register.h>
#include <smith/custom_class.h>
#include <smith/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

int register_vulkan_conv2d_packed_context() {
  static auto register_vulkan_conv2d_context =
      smith::selective_class_<Conv2dPackedContext>(
          "vulkan", SMITH_SELECTIVE_CLASS("Conv2dPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<Conv2dPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<Conv2dPackedContext>(
                    Conv2dPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_conv1d_packed_context() {
  static auto register_vulkan_conv1d_context =
      smith::selective_class_<Conv1dPackedContext>(
          "vulkan", SMITH_SELECTIVE_CLASS("Conv1dPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<Conv1dPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<Conv1dPackedContext>(
                    Conv1dPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_linear_packed_context() {
  static auto register_vulkan_linear_context =
      smith::selective_class_<LinearPackedContext>(
          "vulkan", SMITH_SELECTIVE_CLASS("LinearPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<LinearPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<LinearPackedContext>(
                    LinearPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_layernorm_packed_context() {
  static auto register_vulkan_layernorm_context =
      smith::selective_class_<LayernormPackedContext>(
          "vulkan", SMITH_SELECTIVE_CLASS("LayernormPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<LayernormPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<LayernormPackedContext>(
                    LayernormPackedContext::pack(state));
              });
  return 0;
}

namespace {

SMITH_LIBRARY(vulkan, m) {
  m.class_<BatchNormPackedContext>("BatchNormPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<BatchNormPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<BatchNormPackedContext>(
                BatchNormPackedContext::pack(state));
          });
  m.class_<GruPackedContext>("GruPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GruPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<GruPackedContext>(
                GruPackedContext::pack(state));
          });
  m.class_<LstmPackedContext>("LstmPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<LstmPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<LstmPackedContext>(
                LstmPackedContext::pack(state));
          });
  register_vulkan_conv2d_packed_context();
  register_vulkan_conv1d_packed_context();
  register_vulkan_linear_packed_context();
  register_vulkan_layernorm_packed_context();
  // To maintain backwards compatibility.
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Conv2dOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](Conv2dOpContext::State state) {
            return std::apply(conv2d_clamp_prepack, std::move(state));
          });
}

SMITH_LIBRARY(vulkan_prepack, m) {
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __smith__.smith.classes.vulkan.Conv2dPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __smith__.smith.classes.vulkan.Conv2dOpContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv2d_context(Tensor X, "
      "__smith__.smith.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__smith__.smith.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_tconv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __smith__.smith.classes.vulkan.Conv2dPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_tconv2d_context(Tensor X, "
      "__smith__.smith.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_qconv2d_context(Tensor W, Tensor? B, "
      "int[2] stride, int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __smith__.smith.classes.vulkan.Conv2dPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qconv2d_context(Tensor X, float scale, int zero_point, "
      "__smith__.smith.classes.vulkan.Conv2dPackedContext vk_context) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv1d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups) "
      "-> __smith__.smith.classes.vulkan.Conv1dPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv1d_context(Tensor X, "
      "__smith__.smith.classes.vulkan.Conv1dPackedContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_qtconv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __smith__.smith.classes.vulkan.Conv2dPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_linear_context(Tensor W, Tensor? B) "
      "-> __smith__.smith.classes.vulkan.LinearPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_linear_context(Tensor X, "
      "__smith__.smith.classes.vulkan.LinearPackedContext BW_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qlinear_context(Tensor X, float scale, int zero_point, "
      "__smith__.smith.classes.vulkan.LinearPackedContext vk_context) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_layernorm_context(Tensor? W, Tensor? B, float eps) "
      "-> __smith__.smith.classes.vulkan.LayernormPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_layernorm_context(Tensor X, SymInt[] normalized_shape, "
      "__smith__.smith.classes.vulkan.LayernormPackedContext BW_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_gru_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __smith__.smith.classes.vulkan.GruPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_gru_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "__smith__.smith.classes.vulkan.GruPackedContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_lstm_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __smith__.smith.classes.vulkan.LstmPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_lstm_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "Tensor cx_vk, "
      "__smith__.smith.classes.vulkan.LstmPackedContext L_prepack) -> (Tensor next_input, Tensor hidden_state, Tensor cell_state)"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_batchnorm_context("
      "Tensor? weight_opt, "
      "Tensor? bias_opt, "
      "Tensor? running_mean_opt, "
      "Tensor? running_var_opt, "
      "bool training, "
      "float momentum, "
      "float eps, "
      "bool cudnn_enable) "
      "-> __smith__.smith.classes.vulkan.BatchNormPackedContext"));
  m.def(SMITH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_batchnorm_context("
      "Tensor input_vk, "
      "__smith__.smith.classes.vulkan.BatchNormPackedContext context) "
      "-> Tensor out"));
}

SMITH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_conv2d_context"),
      SMITH_FN(create_conv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"),
      SMITH_FN(conv2d_clamp_prepack)); // Backwards compatibility
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_tconv2d_context"),
      SMITH_FN(create_tconv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_conv1d_context"),
      SMITH_FN(create_conv1d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
      SMITH_FN(create_linear_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_layernorm_context"),
      SMITH_FN(create_layernorm_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_gru_context"),
      SMITH_FN(create_gru_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_lstm_context"),
      SMITH_FN(create_lstm_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_batchnorm_context"),
      SMITH_FN(create_batchnorm_context));
}

SMITH_LIBRARY_IMPL(vulkan_prepack, QuantizedCPU, m) {
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_qconv2d_context"),
      SMITH_FN(create_qconv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::create_qtconv2d_context"),
      SMITH_FN(create_qtconv2d_context));
}

SMITH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_conv2d_context"),
      SMITH_FN(run_conv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"),
      SMITH_FN(conv2d_clamp_run)); // Backwards compatibility
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_tconv2d_context"),
      SMITH_FN(run_tconv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_qconv2d_context"),
      SMITH_FN(run_qconv2d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_conv1d_context"),
      SMITH_FN(run_conv1d_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_linear_context"),
      SMITH_FN(run_linear_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_layernorm_context"),
      SMITH_FN(run_layernorm_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_qlinear_context"),
      SMITH_FN(run_qlinear_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_gru_context"),
      SMITH_FN(run_gru_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_lstm_context"),
      SMITH_FN(run_lstm_context));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_prepack::run_batchnorm_context"),
      SMITH_FN(run_batchnorm_context));
}

SMITH_LIBRARY(vulkan_quantized, m) {
  m.def(
      SMITH_SELECTIVE_SCHEMA("vulkan_quantized::add(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point) -> Tensor qc"));
  m.def(
      SMITH_SELECTIVE_SCHEMA("vulkan_quantized::sub(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  m.def(
      SMITH_SELECTIVE_SCHEMA("vulkan_quantized::mul(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  m.def(
      SMITH_SELECTIVE_SCHEMA("vulkan_quantized::div(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
}

SMITH_LIBRARY_IMPL(vulkan_quantized, Vulkan, m) {
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_quantized::add"), SMITH_FN(quantized_add));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_quantized::sub"), SMITH_FN(quantized_sub));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_quantized::mul"), SMITH_FN(quantized_mul));
  m.impl(
      SMITH_SELECTIVE_NAME("vulkan_quantized::div"), SMITH_FN(quantized_div));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
