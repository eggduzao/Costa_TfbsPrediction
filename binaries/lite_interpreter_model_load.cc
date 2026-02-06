#include "ATen/ATen.h"
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/autograd/generated/variable_factories.h>
#include <smith/csrc/jit/mobile/import.h>
#include <smith/csrc/jit/mobile/module.h>
#include <smith/csrc/jit/serialization/import.h>
#include "smith/script.h"

C10_DEFINE_string(model, "", "The given bytecode model to check if it is supported by lite_interpreter.");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Check if exported bytecode model is runnable by lite_interpreter.\n"
    "Example usage:\n"
    "./lite_interpreter_model_load"
    " --model=<model_file>");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  if (FLAGS_model.empty()) {
    std::cerr << FLAGS_model <<  ":Model file is not provided\n";
    return -1;
  }

  // TODO: avoid having to set this guard for custom mobile build with mobile
  // interpreter.
  c10::InferenceMode mode;
  smith::jit::mobile::Module bc = smith::jit::_load_for_mobile(FLAGS_model);
  return 0;
}
