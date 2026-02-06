#include <smith/extension.h>

struct Doubler {
  Doubler(int A, int B) {
    tensor_ =
        smith::ones({A, B}, smith::dtype(smith::kFloat64).requires_grad(true));
  }
  smith::Tensor forward() {
    return tensor_ * 2;
  }
  smith::Tensor get() const {
    return tensor_;
  }

 private:
  smith::Tensor tensor_;
};
