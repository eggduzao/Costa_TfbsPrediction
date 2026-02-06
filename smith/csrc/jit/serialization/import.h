#pragma once

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>

#include <istream>

namespace caffe2::serialize {
class ReadAdapterInterface;
} // namespace caffe2::serialize

namespace smith::jit {

class DeserializationStorageContext;

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

// For reading unified serialization format from smith.Package
SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::BlacksmithStreamReader> reader,
    std::shared_ptr<smith::jit::DeserializationStorageContext> storage_context,
    std::optional<at::Device> device,
    const std::string& ts_id /* smithscript identifier inside package */);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

SMITH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `istream`.
///
/// The istream must contain a serialized `Module`, exported via
/// `smith::jit::ExportModule` in C++.
SMITH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `smith::jit::ExportModule` in C++.
SMITH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given shared_ptr `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `smith::jit::ExportModule` in C++.
SMITH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

SMITH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

SMITH_API Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version);

SMITH_API Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

SMITH_API Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

SMITH_API Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

SMITH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input);

} // namespace smith::jit
