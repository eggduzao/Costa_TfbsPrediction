#include <ATen/UnboxingFunctions.h>
#include <ATen/Functions.h>

#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace at {
namespace unboxing {

using ::c10::fmap;
using ::c10::filter;
using smith::jit::peek;
using smith::jit::drop;
using smith::jit::pack;
using smith::jit::pop;

// Generated function declaration
${definitions}

} // namespace unboxing
} // namespace at
