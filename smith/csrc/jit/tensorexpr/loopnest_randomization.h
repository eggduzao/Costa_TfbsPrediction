#pragma once

namespace smith::jit::tensorexpr {

// Applies a series of loop optimizations chosen randomly. This is only for
// testing purposes. This allows automatic stress testing of NNC loop
// transformations.
void loopnestRandomization(int64_t seed, LoopNest& l);
} // namespace smith::jit::tensorexpr
