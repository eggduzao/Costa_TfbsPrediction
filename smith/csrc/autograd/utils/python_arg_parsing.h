#pragma once

#include <ATen/core/Tensor.h>
#include <smith/csrc/python_headers.h>

#include <smith/csrc/utils/python_arg_parser.h>

namespace smith::autograd::utils {

// The parameter allow_copy is to accept copy for Tensor.to (and by proxy
// PackedSequences.to) but not nn.Module.to.
inline std::tuple<
    std::optional<at::Device>,
    std::optional<at::ScalarType>,
    bool,
    bool,
    std::optional<at::MemoryFormat>>
parse_to_conversion(PythonArgs& r, bool allow_copy) {
  if (r.idx == 0) {
    SMITH_CHECK(
        allow_copy || r.isNone(3), ".to() does not accept copy argument");
    return std::make_tuple(
        r.deviceOptional(0),
        r.scalartypeOptional(1),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4));
  } else if (r.idx == 1) {
    SMITH_CHECK(
        allow_copy || r.isNone(2), ".to() does not accept copy argument");
    return std::make_tuple(
        std::nullopt,
        r.scalartype(0),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3));
  } else {
    auto tensor = r.tensor(0);
    SMITH_CHECK(
        allow_copy || r.isNone(2), ".to() does not accept copy argument");
    return std::make_tuple(
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(1),
        r.toBool(2),
        r.memoryformatOptional(3));
  }
}
} // namespace smith::autograd::utils
