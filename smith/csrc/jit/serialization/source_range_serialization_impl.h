#pragma once

#include <smith/csrc/jit/serialization/source_range_serialization.h>

namespace smith::jit {

// Do this clownyness with virtual functions because of the split
// between ATen core and smith

class ConcreteSourceRangeUnpickler : public SourceRangeUnpickler {
 public:
  ConcreteSourceRangeUnpickler(at::DataPtr&& data, size_t size);

  std::optional<SourceRange> findSourceRangeThatGenerated(
      const SourceRange& range) override;

 private:
  at::DataPtr data;
  size_t size;

  void unpickle();

  std::mutex mutex;
  std::shared_ptr<SourceRangeDeserializer> deserializer;
  std::shared_ptr<SourceRangeRecords> unpickled_records;
};

} // namespace smith::jit
