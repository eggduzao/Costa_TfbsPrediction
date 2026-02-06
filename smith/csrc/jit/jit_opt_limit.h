#pragma once
#include <smith/csrc/Export.h>
#include <string>
#include <unordered_map>

// `SmithScript` offers a simple optimization limit checker
// that can be configured through environment variable `BLACKSMITH_JIT_OPT_LIMIT`.
// The purpose is to limit how many optimization you can make per pass.
// This is useful for debugging any passes.

// Opt limit checker is enabled on a per file basis (hence per pass). For
// example, in `constant_propagation.cpp`, `BLACKSMITH_JIT_OPT_LIMIT` should be set
// to `constant_propagation=<opt_limit>` or, simply, to
// `constant_propagation=<opt_limit>` where <opt_limit> is the number of
// optimizations you want to make for the pass. (i.e.
// `BLACKSMITH_JIT_OPT_LIMIT="constant_propagation=<opt_limit>"`).

// Multiple files can be configured by separating each file name with a colon
// `:` as in the following example,
// `BLACKSMITH_JIT_OPT_LIMIT="constant_propagation=<opt_limit>:dead_code_elimination=<opt_limit>"`

// You can call opt limiter by calling JIT_OPT_ALLOWED. It will return true if
// we haven't reached the optimization limit yet. Otherwise, it will return
// false. Typical usage:

// if (!JIT_OPT_ALLOWED) {
//     GRAPH_DUMP(...); //supplied from jit_log
//     return;
// }

namespace smith::jit {

SMITH_API bool opt_limit(const char* pass_name);

#define JIT_OPT_ALLOWED opt_limit(__FILE__)

} // namespace smith::jit
