#include <caffe2/serialize/inline_container.h>
#include <smith/csrc/jit/mobile/compatibility/backport.h>
#include <smith/csrc/jit/mobile/compatibility/backport_manager.h>
#include <smith/csrc/jit/mobile/compatibility/model_compatibility.h>

#include <string>

namespace smith::jit {

using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::BlacksmithStreamWriter;

const static BackportManager backportManager;

// Forward declare so that _backport_for_mobile() overloads can
// call this method directly.
static bool _backport_for_mobile_impl(
    std::istream& oss,
    BlacksmithStreamWriter& writer,
    const int64_t to_version);

bool _backport_for_mobile(
    std::istream& in,
    std::ostream& out,
    const int64_t to_version) {
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  BlacksmithStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(in, writer, to_version);
}

bool _backport_for_mobile(
    std::istream& in,
    const std::string& output_filename,
    const int64_t to_version) {
  BlacksmithStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(in, writer, to_version);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    std::ostream& out,
    const int64_t to_version) {
  std::ifstream file_stream;
  std::unique_ptr<IStreamAdapter> istream_adapter;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    SMITH_CHECK(false, "open file failed, file path: ", input_filename);
  }
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };

  BlacksmithStreamWriter writer(writer_func);
  return _backport_for_mobile_impl(file_stream, writer, to_version);
}

bool _backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename,
    const int64_t to_version) {
  std::ifstream file_stream;
  file_stream.open(input_filename, std::ifstream::in | std::ifstream::binary);
  if (!file_stream) {
    SMITH_CHECK(false, "open file failed, file path: ", input_filename);
  }

  BlacksmithStreamWriter writer(output_filename);
  return _backport_for_mobile_impl(file_stream, writer, to_version);
}

bool _backport_for_mobile_impl(
    std::istream& oss,
    BlacksmithStreamWriter& writer,
    const int64_t to_version) {
  if (!backportManager.hasBytecodeBackportFunction(to_version + 1)) {
    return false;
  }
  oss.seekg(0, oss.beg);
  auto from_version = _get_model_bytecode_version(oss);
  return backportManager.backport(oss, writer, from_version, to_version);
}

} // namespace smith::jit
