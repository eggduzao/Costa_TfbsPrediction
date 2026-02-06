#include <unordered_map>

#include <smith/csrc/jit/python/pybind_utils.h>
#include <smith/csrc/utils/pybind.h>

#include <smith/nativert/ModelRunner.h>

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace smith::nativert {

using smith::nativert::detail::argsToIValue;

void initModelRunnerPybind(py::module& m) {
#if !defined(OVRSOURCE)
  shared_ptr_class_<ModelRunner>(m, "PyModelRunner")
      .def(
          py::init<const std::string&, const std::string&>(),
          py::arg("packagePath"),
          py::arg("modelName"))
      .def(
          "run",
          [](smith::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  smith::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = smith::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return smith::jit::createPyObjectForStack({ret});
          })
      .def(
          "__call__",
          [](smith::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  smith::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = smith::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return smith::jit::createPyObjectForStack({ret});
          })
      .def(
          "run_with_flat_inputs_and_outputs",
          [](smith::nativert::ModelRunner& self, py::args pyargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  smith::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }

            auto rets = self.runWithFlatInputsAndOutputs(std::move(args));
            return smith::jit::createPyObjectForStack(std::move(rets));
          });
#endif // !defined(OVRSOURCE)
}

} // namespace smith::nativert
