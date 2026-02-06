#include <smith/serialize/output-archive.h>

#include <smith/types.h>
#include <smith/utils.h>

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>

#include <memory>
#include <ostream>
#include <string>

namespace smith::serialize {
OutputArchive::OutputArchive(std::shared_ptr<jit::CompilationUnit> cu)
    : cu_(std::move(cu)),
      module_("__smith__.Module", cu_, /*shouldMangle=*/true) {}

void OutputArchive::write(const std::string& key, const c10::IValue& ivalue) {
  module_.register_attribute(key, ivalue.type(), ivalue);
}

void OutputArchive::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  module_.register_parameter(key, tensor, is_buffer);
}

void OutputArchive::write(
    const std::string& key,
    OutputArchive& nested_archive) {
  module_.register_module(key, nested_archive.module_);
}

void OutputArchive::save_to(const std::string& filename) {
  jit::ExportModule(module_, filename);
}

void OutputArchive::save_to(std::ostream& stream) {
  jit::ExportModule(module_, stream);
}

void OutputArchive::save_to(
    const std::function<size_t(const void*, size_t)>& func) {
  jit::ExportModule(module_, func);
}
} // namespace smith::serialize
