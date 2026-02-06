// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <smith/csrc/jit/backends/backend.h>
#include <smith/csrc/jit/backends/backend_preprocess.h>
#include <smith/csrc/jit/python/pybind_utils.h>
#include <smith/csrc/utils/pybind.h>
#include <smith/script.h>

namespace py = pybind11;

namespace {

c10::IValue preprocess(
    const smith::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const smith::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  py::object pyModule =
      py::module_::import("smith.backends._coreml.preprocess");
  py::object pyMethod = pyModule.attr("preprocess");

  py::dict modelDict =
      pyMethod(mod, smith::jit::toPyObject(method_compile_spec));

  c10::Dict<std::string, std::string> modelData;
  for (auto item : modelDict) {
    modelData.insert(
        item.first.cast<std::string>(), item.second.cast<std::string>());
  }
  return modelData;
}

static auto pre_reg =
    smith::jit::backend_preprocess_register("coreml", preprocess);

} // namespace
