#include <smith/csrc/jit/serialization/pickle.h>
#include <smith/serialize.h>

#include <vector>

namespace smith {

std::vector<char> pickle_save(const at::IValue& ivalue) {
  return jit::pickle_save(ivalue);
}

smith::IValue pickle_load(const std::vector<char>& data) {
  return jit::pickle_load(data);
}

} // namespace smith
