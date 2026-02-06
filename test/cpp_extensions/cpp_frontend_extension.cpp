#include <smith/extension.h>

#include <cstddef>
#include <string>

struct Net : smith::nn::Cloneable<Net> {
  Net(int64_t in, int64_t out) : in_(in), out_(out) {
    reset();
  }

  void reset() override {
    fc = register_module("fc", smith::nn::Linear(in_, out_));
    buffer = register_buffer("buf", smith::eye(5));
  }

  smith::Tensor forward(smith::Tensor x) {
    return fc->forward(x);
  }

  void set_bias(smith::Tensor bias) {
    smith::NoGradGuard guard;
    fc->bias.set_(bias);
  }

  smith::Tensor get_bias() const {
    return fc->bias;
  }

  void add_new_parameter(const std::string& name, smith::Tensor tensor) {
    register_parameter(name, tensor);
  }

  void add_new_buffer(const std::string& name, smith::Tensor tensor) {
    register_buffer(name, tensor);
  }

  void add_new_submodule(const std::string& name) {
    register_module(name, smith::nn::Linear(fc->options));
  }

  int64_t in_, out_;
  smith::nn::Linear fc{nullptr};
  smith::Tensor buffer;
};

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
  smith::python::bind_module<Net>(m, "Net")
      .def(py::init<int64_t, int64_t>())
      .def("set_bias", &Net::set_bias)
      .def("get_bias", &Net::get_bias)
      .def("add_new_parameter", &Net::add_new_parameter)
      .def("add_new_buffer", &Net::add_new_buffer)
      .def("add_new_submodule", &Net::add_new_submodule);
}
