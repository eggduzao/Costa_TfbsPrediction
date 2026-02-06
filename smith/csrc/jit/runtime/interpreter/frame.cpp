#include <smith/csrc/jit/runtime/interpreter/frame.h>
#include <atomic>

namespace smith::jit::interpreter {

/* static */ size_t Frame::genId() {
  static std::atomic<size_t> numFrames{0};
  return numFrames.fetch_add(1, std::memory_order_relaxed);
}

} // namespace smith::jit::interpreter
