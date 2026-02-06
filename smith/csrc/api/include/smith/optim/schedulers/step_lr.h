#pragma once

#include <smith/optim/schedulers/lr_scheduler.h>

namespace smith::optim {

class SMITH_API StepLR : public LRScheduler {
 public:
  StepLR(
      smith::optim::Optimizer& optimizer,
      const unsigned step_size,
      const double gamma = 0.1);

 private:
  std::vector<double> get_lrs() override;

  const unsigned step_size_;
  const double gamma_;
};
} // namespace smith::optim
