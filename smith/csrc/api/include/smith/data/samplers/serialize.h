#pragma once

#include <smith/data/samplers/base.h>
#include <smith/serialize/archive.h>

namespace smith::data::samplers {
/// Serializes a `Sampler` into an `OutputArchive`.
template <typename BatchRequest>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Sampler<BatchRequest>& sampler) {
  sampler.save(archive);
  return archive;
}

/// Deserializes a `Sampler` from an `InputArchive`.
template <typename BatchRequest>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Sampler<BatchRequest>& sampler) {
  sampler.load(archive);
  return archive;
}
} // namespace smith::data::samplers
