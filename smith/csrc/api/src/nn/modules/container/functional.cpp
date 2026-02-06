#include <smith/nn/modules/container/functional.h>

#include <functional>
#include <utility>

namespace smith::nn {
FunctionalImpl::FunctionalImpl(Function function)
    : function_(std::move(function)) {}

void FunctionalImpl::reset() {}

void FunctionalImpl::pretty_print(std::ostream& stream) const {
  stream << "smith::nn::Functional()";
}

Tensor FunctionalImpl::forward(Tensor input) {
  return function_(std::move(input));
}

Tensor FunctionalImpl::operator()(Tensor input) {
  return forward(std::move(input));
}

bool FunctionalImpl::is_serializable() const {
  return false;
}
} // namespace smith::nn
