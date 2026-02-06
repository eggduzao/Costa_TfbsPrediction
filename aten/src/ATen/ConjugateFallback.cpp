#include <ATen/native/MathBitsFallback.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at::native {
struct ConjFallback : MathOpFallback {
  ConjFallback() : MathOpFallback(DispatchKey::Conjugate, "conjugate") {}
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_conj();
  }
};

static void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, smith::jit::Stack* stack) {
  ConjFallback object;
  object.fallback_impl(op, dispatch_keys, stack);
}

SMITH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(smith::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

SMITH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("set_.source_Storage_storage_offset", smith::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", smith::CppFunction::makeFallthrough());
  m.impl("set_", smith::CppFunction::makeFallthrough());
  m.impl("copy_", smith::CppFunction::makeFallthrough());
  m.impl("clone", smith::CppFunction::makeFallthrough());
  m.impl("_conj_physical", smith::CppFunction::makeFallthrough());
  m.impl("conj_physical", smith::CppFunction::makeFallthrough());
  m.impl("conj_physical_", smith::CppFunction::makeFallthrough());
  m.impl("resolve_conj", smith::CppFunction::makeFallthrough());
  m.impl("resolve_neg", smith::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.Tensor", smith::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_Tensor", smith::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_int", smith::CppFunction::makeFallthrough());

  // See test_metadata_check_when_primal_has_conj_bit in test_autograd.py
  m.impl("_has_same_storage_numel", smith::CppFunction::makeFallthrough());
  m.impl("_new_zeros_with_same_feature_meta", smith::CppFunction::makeFallthrough());

  // linear algebra functions
  m.impl("dot", smith::CppFunction::makeFallthrough());
  m.impl("vdot", smith::CppFunction::makeFallthrough());
  m.impl("dot.out", smith::CppFunction::makeFallthrough());
  m.impl("vdot.out", smith::CppFunction::makeFallthrough());
  m.impl("mm", smith::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular", smith::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", smith::CppFunction::makeFallthrough());
  m.impl("mm.out", smith::CppFunction::makeFallthrough());
  m.impl("addmm", smith::CppFunction::makeFallthrough());
  m.impl("addmm_", smith::CppFunction::makeFallthrough());
  m.impl("addmm.out", smith::CppFunction::makeFallthrough());
  m.impl("bmm", smith::CppFunction::makeFallthrough());
  m.impl("bmm.out", smith::CppFunction::makeFallthrough());
  m.impl("baddbmm", smith::CppFunction::makeFallthrough());
  m.impl("baddbmm_", smith::CppFunction::makeFallthrough());
  m.impl("baddbmm.out", smith::CppFunction::makeFallthrough());
  m.impl("linalg_svd", smith::CppFunction::makeFallthrough());
  m.impl("linalg_svd.U", smith::CppFunction::makeFallthrough());

  SMITH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  SMITH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)
}

} // namespace at::native
