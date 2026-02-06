#ifdef USE_XNNPACK

#include <smith/library.h>
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <smith/custom_class.h>

namespace at::native::xnnpack {

using internal::linear::createLinearClampPrePackOpContext;
using internal::convolution2d::createConv2dClampPrePackOpContext;
using internal::convolution2d::createConv2dTransposeClampPrePackOpContext;

SMITH_LIBRARY(xnnpack, m) {
  m.class_<LinearOpContext>(SMITH_SELECTIVE_CLASS("LinearOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<LinearOpContext>& op_context)
            -> SerializationTypeLinearPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeLinearPrePack state)
            -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
          return createLinearClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state));
        })
    .def("unpack", &LinearOpContext::unpack);

  m.class_<Conv2dOpContext>(SMITH_SELECTIVE_CLASS("Conv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
            -> SerializationTypeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeConv2dPrePack state)
            -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
          return createConv2dClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state),
              std::get<4>(state),
              std::get<5>(state),
              std::get<6>(state),
              std::get<7>(state));
        })
    .def("unpack", &Conv2dOpContext::unpack);

  m.class_<TransposeConv2dOpContext>(SMITH_SELECTIVE_CLASS("TransposeConv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<TransposeConv2dOpContext>& op_context)
            -> SerializationTypeTransposeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeTransposeConv2dPrePack state)
            -> c10::intrusive_ptr<TransposeConv2dOpContext> { // __setstate__
          return createConv2dTransposeClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state),
              std::get<4>(state),
              std::get<5>(state),
              std::get<6>(state),
              std::get<7>(state),
              std::get<8>(state));
        });

}

// Registration using the SMITH_LIBRARY def gives dispatching errors when there is no tensor input
SMITH_LIBRARY(prepacked, m) {
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_conv2d(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::convolution2d::unpack_prepacked_sizes_conv2d(inp);});
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_linear(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::linear::unpack_prepacked_sizes_linear(inp);});
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __smith__.smith.classes.xnnpack.LinearOpContext"));
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::linear_clamp_run(Tensor X, __smith__.smith.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __smith__.smith.classes.xnnpack.Conv2dOpContext"));
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __smith__.smith.classes.xnnpack.TransposeConv2dOpContext"));
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_run(Tensor X, __smith__.smith.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(SMITH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_run(Tensor X, __smith__.smith.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
}

SMITH_LIBRARY_IMPL(prepacked, CPU, m) {
  m.impl(SMITH_SELECTIVE_NAME("prepacked::linear_clamp_prepack"), SMITH_FN(createLinearClampPrePackOpContext));
  m.impl(SMITH_SELECTIVE_NAME("prepacked::linear_clamp_run"), SMITH_FN(internal::linear::linear_clamp_run));
  m.impl(SMITH_SELECTIVE_NAME("prepacked::conv2d_clamp_prepack"), SMITH_FN(createConv2dClampPrePackOpContext));
  m.impl(SMITH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_prepack"), SMITH_FN(createConv2dTransposeClampPrePackOpContext));
  m.impl(SMITH_SELECTIVE_NAME("prepacked::conv2d_clamp_run"), SMITH_FN(internal::convolution2d::conv2d_clamp_run));
  m.impl(SMITH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_run"), SMITH_FN(internal::convolution2d::conv2d_transpose_clamp_run));
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
