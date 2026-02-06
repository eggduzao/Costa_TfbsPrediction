#include <smith/csrc/jit/mobile/observer.h>

namespace smith {

MobileObserverConfig& observerConfig() {
  static MobileObserverConfig instance;
  return instance;
}

} // namespace smith
