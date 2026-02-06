#pragma once

#include <smith/csrc/jit/frontend/source_range.h>
#include <smith/csrc/jit/frontend/tracer.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/pybind.h>

#include <memory>
#include <string>

namespace smith::jit {

struct Module;

namespace tracer {
void initPythonTracerBindings(PyObject* module);

SourceRange getPythonInterpreterSourceRange();

Node* preRecordPythonTrace(
    THPObjectPtr pyobj,
    const std::string& arg_types,
    at::ArrayRef<autograd::Variable> inputs,
    std::vector<THPObjectPtr> scalar_args);

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracingWithDict(
    const py::function& func,
    const py::dict& inputs_dict,
    const Stack& inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});

std::pair<std::shared_ptr<Graph>, Stack> createGraphByTracing(
    const py::function& func,
    Stack inputs,
    const py::function& var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});
} // namespace tracer
} // namespace smith::jit
