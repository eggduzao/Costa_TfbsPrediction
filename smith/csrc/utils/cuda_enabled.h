#pragma once

namespace smith::utils {

inline constexpr bool cuda_enabled() {
#ifdef USE_CUDA
  return true;
#else
  return false;
#endif
}

} // namespace smith::utils
