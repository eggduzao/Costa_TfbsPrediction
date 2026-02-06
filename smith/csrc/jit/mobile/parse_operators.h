#pragma once
#include <smith/csrc/jit/mobile/function.h>

namespace smith::jit {
using c10::IValue;

enum MobileModuleLoadOptions {
  OPERASMITHECK = 1,
  // PARSE_ALL_EXTRA_FILE_MAPS is used to gate for ExtraFileMaps to pull all
  // files automatically without explicit entries mapping. Refer to PR for a
  // detail: https://github.com/blacksmith/blacksmith/pull/99747
  PARSE_ALL_EXTRA_FILE_MAPS = 2,
};

const uint64_t kDefaultMobileLoadOptions =
    MobileModuleLoadOptions::OPERASMITHECK;

namespace mobile {

SMITH_API void parseOperators(
    c10::ivalue::TupleElements&& ops_list,
    const uint64_t& module_load_options,
    mobile::Function* function);
} // namespace mobile
} // namespace smith::jit
