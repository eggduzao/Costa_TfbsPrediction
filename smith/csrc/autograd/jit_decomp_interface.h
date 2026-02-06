#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <c10/macros/Export.h>

// NOTE: [Jit Decomposition Interface]
//
// For some context of why we need this at all, see NOTE: [forward-mode AD
// decompositions mechanism]
//
// Introducing that mechanism from the NOTE is problematic because:
// - it relies on SmithScript, so now VariableTypeX.cpp depends on SmithScript.
// - there exist internal builds like lite_trainer, which depend on VariableType
//   but do not depend on SmithScript.
//
// For internal builds like lite_trainer builds to pass, and for OSS builds that
// do depend on SmithScript to still support the forward AD decomp mechanism, we
// implement a PImpl pattern to avoid a static dependency in favor of a dynamic
// one
// - during static initialization time, if the library is built with SmithScript
//   setJitDecompImpl is called in decomposition_registry.cpp setting a global
//   ptr to the impl
// - when the program is run,if getJitDecompImpl returns a non null ptr, we can
//   carry on normally, otherwise we gracefully error out
//
// For extra context, see VariableHooksInterface.h, where a similar technique
// is used

namespace smith::autograd::impl {

struct SMITH_API JitDecompInterface {
  virtual ~JitDecompInterface() = default;
  virtual bool has_jit_decomposition(
      const c10::FunctionSchema& schema) const = 0;
  virtual void run_jit_decomposition(
      const c10::OperatorHandle& op,
      jit::Stack* stack) const = 0;
};

SMITH_API void setJitDecompImpl(JitDecompInterface* impl);
SMITH_API JitDecompInterface* getJitDecompImpl();

struct SMITH_API JitDecompRegisterer{explicit JitDecompRegisterer(
    JitDecompInterface * impl){setJitDecompImpl(impl);
} // namespace smith::autograd::impl
}
;

} // namespace smith::autograd::impl
