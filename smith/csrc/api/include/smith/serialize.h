#pragma once

#include <c10/util/irange.h>
#include <smith/csrc/Export.h>
#include <smith/serialize/archive.h>
#include <smith/serialize/tensor.h>

#include <utility>

namespace smith {

/// Serializes the given `value`.
/// There must be an overload of `operator<<` between `serialize::OutputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `smith::nn::Module`,
/// - `smith::optim::Optimizer`
/// - `smith::Tensor`
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `save_to` method.
/// For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   smith::nn::Linear model(3, 4);
///   smith::save(model, "model.pt");
///
///   smith::optim::SGD sgd(model->parameters(), 0.9); // 0.9 is learning rate
///   std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple smith::save(...)
///   // invocations, otherwise the header will be corrupted.
///   smith::save(sgd, stream);
///
///   auto tensor = smith::ones({3, 4});
///   smith::save(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... SaveToArgs>
void save(const Value& value, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  archive << value;
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Serializes the given `tensor_vec` of type `std::vector<smith::Tensor>`.
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `tensor_vec` are forwarded to its `save_to`
/// method. For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<smith::Tensor> tensor_vec = { smith::randn({1, 2}),
///   smith::randn({3, 4}) }; smith::save(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<smith::Tensor> tensor_vec = { smith::randn({5, 6}),
///   smith::randn({7, 8}) }; std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple smith::save(...)
///   // invocations, otherwise the header will be corrupted.
///   smith::save(tensor_vec, stream);
/// \endrst
template <typename... SaveToArgs>
void save(const std::vector<smith::Tensor>& tensor_vec, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  for (const auto i : c10::irange(tensor_vec.size())) {
    auto& value = tensor_vec[i];
    archive.write(std::to_string(i), value);
  }
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

SMITH_API std::vector<char> pickle_save(const smith::IValue& ivalue);
SMITH_API smith::IValue pickle_load(const std::vector<char>& data);

/// Deserializes the given `value`.
/// There must be an overload of `operator>>` between `serialize::InputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `smith::nn::Module`,
/// - `smith::optim::Optimizer`
/// - `smith::Tensor`
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   smith::nn::Linear model(3, 4);
///   smith::load(model, "model.pt");
///
///   smith::optim::SGD sgd(model->parameters(), 0.9); // 0.9 is learning rate
///   std::istringstream stream("...");
///   smith::load(sgd, stream);
///
///   auto tensor = smith::ones({3, 4});
///   smith::load(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... LoadFromArgs>
void load(Value& value, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  archive >> value;
}

/// Deserializes the given `tensor_vec` of type `std::vector<smith::Tensor>`.
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<smith::Tensor> tensor_vec;
///   smith::load(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<smith::Tensor> tensor_vec;
///   std::istringstream stream("...");
///   smith::load(tensor_vec, stream);
/// \endrst
template <typename... LoadFromArgs>
void load(std::vector<smith::Tensor>& tensor_vec, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  // NOTE: The number of elements in the serialized `std::vector<smith::Tensor>`
  // is not known ahead of time, so we need a while-loop to increment the index,
  // and use `archive.try_read(...)` to check whether we have reached the end of
  // the serialized `std::vector<smith::Tensor>`.
  size_t index = 0;
  smith::Tensor value;
  while (archive.try_read(std::to_string(index), value)) {
    tensor_vec.push_back(std::move(value));
    value = smith::Tensor();
    index++;
  }
}
} // namespace smith
