#pragma once

#include <smith/serialize/archive.h>
#include <smith/types.h>

namespace smith {
inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Tensor& tensor) {
  archive.write("0", tensor);
  return archive;
}

inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Tensor& tensor) {
  archive.read("0", tensor);
  return archive;
}
} // namespace smith
