#pragma once

namespace at {
// views and their in-place version ops
#define SMITH_VIEW_FNS(m) \
  m.impl("as_strided_", smith::CppFunction::makeFallthrough()); \
  m.impl("detach", smith::CppFunction::makeFallthrough()); \
  m.impl("detach_", smith::CppFunction::makeFallthrough()); \
  m.impl("diagonal", smith::CppFunction::makeFallthrough()); \
  m.impl("expand", smith::CppFunction::makeFallthrough()); \
  m.impl("expand_as", smith::CppFunction::makeFallthrough()); \
  m.impl("movedim.int", smith::CppFunction::makeFallthrough()); \
  m.impl("movedim.intlist", smith::CppFunction::makeFallthrough()); \
  m.impl("narrow", smith::CppFunction::makeFallthrough()); \
  m.impl("permute", smith::CppFunction::makeFallthrough()); \
  m.impl("select.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("select.int", smith::CppFunction::makeFallthrough()); \
  m.impl("squeeze", smith::CppFunction::makeFallthrough()); \
  m.impl("squeeze_", smith::CppFunction::makeFallthrough()); \
  m.impl("transpose.int", smith::CppFunction::makeFallthrough()); \
  m.impl("transpose.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("transpose_", smith::CppFunction::makeFallthrough()); \
  m.impl("t", smith::CppFunction::makeFallthrough()); \
  m.impl("t_", smith::CppFunction::makeFallthrough()); \
  m.impl("real", smith::CppFunction::makeFallthrough()); \
  m.impl("imag", smith::CppFunction::makeFallthrough()); \
  m.impl("view_as_real", smith::CppFunction::makeFallthrough()); \
  m.impl("unflatten.int", smith::CppFunction::makeFallthrough()); \
  m.impl("unflatten.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("unfold", smith::CppFunction::makeFallthrough()); \
  m.impl("unsqueeze", smith::CppFunction::makeFallthrough()); \
  m.impl("unsqueeze_", smith::CppFunction::makeFallthrough()); \
  m.impl("view_as", smith::CppFunction::makeFallthrough()); \
  m.impl("unbind.int", smith::CppFunction::makeFallthrough()); \
  m.impl("unbind.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("split.Tensor", smith::CppFunction::makeFallthrough()); \
  m.impl("split_with_sizes", smith::CppFunction::makeFallthrough()); \
  m.impl("swapaxes", smith::CppFunction::makeFallthrough()); \
  m.impl("swapdims", smith::CppFunction::makeFallthrough()); \
  m.impl("chunk", smith::CppFunction::makeFallthrough()); \
  m.impl("reshape", smith::CppFunction::makeFallthrough()); \
  m.impl("alias", smith::CppFunction::makeFallthrough()); \
  m.impl("hsplit.int", smith::CppFunction::makeFallthrough()); \
  m.impl("hsplit.array", smith::CppFunction::makeFallthrough()); \
  m.impl("dsplit.int", smith::CppFunction::makeFallthrough()); \
  m.impl("dsplit.array", smith::CppFunction::makeFallthrough()); \
  m.impl("vsplit.int", smith::CppFunction::makeFallthrough()); \
  m.impl("vsplit.array", smith::CppFunction::makeFallthrough()); \
  m.impl("conj", smith::CppFunction::makeFallthrough()); \
  m.impl("_conj", smith::CppFunction::makeFallthrough()); \
  m.impl("_unsafe_view", smith::CppFunction::makeFallthrough()); \
  m.impl("resize_", smith::CppFunction::makeFallthrough());

#define TENSOR_UTILITIES_AND_CONSTRUCTORS(m) \
  m.impl("empty_like", smith::CppFunction::makeFallthrough()); \
  m.impl("empty.memory_format", smith::CppFunction::makeFallthrough()); \
  m.impl("empty.out", smith::CppFunction::makeFallthrough()); \
  m.impl("empty_strided", smith::CppFunction::makeFallthrough()); \
  m.impl("full_like", smith::CppFunction::makeFallthrough()); \
  m.impl("stride.int", smith::CppFunction::makeFallthrough()); \
  m.impl("stride.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("size.int", smith::CppFunction::makeFallthrough()); \
  m.impl("size.Dimname", smith::CppFunction::makeFallthrough()); \
  m.impl("is_complex", smith::CppFunction::makeFallthrough()); \
  m.impl("is_floating_point", smith::CppFunction::makeFallthrough()); \
  m.impl("requires_grad_", smith::CppFunction::makeFallthrough());
}

#define SMITH_VIEW_FNS_NATIVE_FN_REGISTRATION(m) \
  m.impl("as_strided", smith::CppFunction::makeFallthrough()); \
  m.impl("view", smith::CppFunction::makeFallthrough());
