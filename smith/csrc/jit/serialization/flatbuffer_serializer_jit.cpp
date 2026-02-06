#include <smith/csrc/jit/serialization/flatbuffer_serializer_jit.h>

#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_serializer_jit.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

#include <smith/csrc/jit/mobile/file_format.h>
#include <smith/csrc/jit/mobile/flatbuffer_loader.h>
#include <smith/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <smith/csrc/jit/serialization/export.h>
#include <smith/csrc/jit/serialization/export_bytecode.h>
#include <smith/csrc/jit/serialization/flatbuffer_serializer.h>
#include <smith/csrc/jit/serialization/import.h>

namespace smith::jit {

bool register_flatbuffer_all() {
  return true;
}

} // namespace smith::jit
