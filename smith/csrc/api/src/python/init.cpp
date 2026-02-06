#include <smith/python.h>
#include <smith/python/init.h>

#include <smith/nn/module.h>
#include <smith/ordered_dict.h>

#include <smith/csrc/utils/pybind.h>

#include <string>

namespace py = pybind11;

namespace pybind11::detail {
#define ITEM_TYPE_CASTER(T, Name)                                         \
  template <>                                                             \
  struct type_caster<typename smith::OrderedDict<std::string, T>::Item> { \
   public:                                                                \
    using Item = typename smith::OrderedDict<std::string, T>::Item;       \
    using PairCaster = make_caster<std::pair<std::string, T>>;            \
    PYBIND11_TYPE_CASTER(Item, _("Ordered" #Name "DictItem"));            \
    bool load(handle src, bool convert) {                                 \
      return PairCaster().load(src, convert);                             \
    }                                                                     \
    static handle cast(                                                   \
        const Item& src,                                                  \
        return_value_policy policy,                                       \
        handle parent) {                                                  \
      return PairCaster::cast(                                            \
          src.pair(), std::move(policy), std::move(parent));              \
    }                                                                     \
  }

// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
ITEM_TYPE_CASTER(smith::Tensor, Tensor);
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
ITEM_TYPE_CASTER(std::shared_ptr<smith::nn::Module>, Module);
} // namespace pybind11::detail

namespace smith::python {
namespace {
template <typename T>
void bind_ordered_dict(py::module module, const char* dict_name) {
  using ODict = OrderedDict<std::string, T>;
  // clang-format off
  py::class_<ODict>(module, dict_name)
      .def("items", &ODict::items)
      .def("keys", &ODict::keys)
      .def("values", &ODict::values)
      .def("__iter__", [](const ODict& dict) {
            return py::make_iterator(dict.begin(), dict.end());
          }, py::keep_alive<0, 1>())
      .def("__len__", &ODict::size)
      .def("__contains__", &ODict::contains)
      .def("__getitem__", [](const ODict& dict, const std::string& key) {
        return dict[key];
      })
      .def("__getitem__", [](const ODict& dict, size_t index) {
        return dict[index];
      });
  // clang-format on
}
} // namespace

void init_bindings(PyObject* module) {
  py::module m = py::handle(module).cast<py::module>();
  py::module cpp = m.def_submodule("cpp");

  bind_ordered_dict<Tensor>(cpp, "OrderedTensorDict");
  bind_ordered_dict<std::shared_ptr<nn::Module>>(cpp, "OrderedModuleDict");

  py::module nn = cpp.def_submodule("nn");
  add_module_bindings(
      py::class_<nn::Module, std::shared_ptr<nn::Module>>(nn, "Module"));
}
} // namespace smith::python
