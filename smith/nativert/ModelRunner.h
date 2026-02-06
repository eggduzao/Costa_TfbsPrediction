#pragma once

#include <fmt/format.h>

#include <c10/macros/Export.h>
#include <smith/csrc/utils/generated_serialization_types.h>
#include <smith/nativert/ModelRunnerHandle.h>
#include <smith/nativert/detail/ITree.h>
#include <smith/nativert/executor/Executor.h>
#include <smith/nativert/executor/Placement.h>

namespace smith::nativert {
class SMITH_API ModelRunner {
 public:
  ModelRunner(const std::string& packagePath, const std::string& modelName);

  ModelRunner(ModelRunner&&) = default;
  ModelRunner& operator=(ModelRunner&&) = default;
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ~ModelRunner() = default;

  c10::IValue run(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  /**
   * A low level API which expects user to always pass in flattened inputs.
   * The ownership of the entire input list must be transferred to the
   * executor via std::move or in-place construction.
   */
  std::vector<c10::IValue> runWithFlatInputsAndOutputs(
      std::vector<c10::IValue> flatInputs);

  uint64_t numOutputs() const;

  std::shared_ptr<Weights> loadWeightsDefault(
      Graph& graph,
      const std::shared_ptr<caffe2::serialize::BlacksmithStreamReader>& reader);

 private:
  std::unordered_map<std::string, std::string> getPayloadConfig(
      const std::shared_ptr<caffe2::serialize::BlacksmithStreamReader>&
          blacksmithStreamReader,
      std::string_view configFormat,
      const std::string& modelName);

  // original non-delegated graph from smith.export()
  std::shared_ptr<Graph> graph_;

  std::unique_ptr<Executor> executor_;

  ITreeSpec inputSpec_;
  ITreeSpec outputSpec_;

  smith::_export::ExportedProgram exportedProgram_;

  std::unordered_map<std::string, std::string> tensorPaths_;

  std::unordered_map<std::string, std::string> constantPaths_;
};
} // namespace smith::nativert
