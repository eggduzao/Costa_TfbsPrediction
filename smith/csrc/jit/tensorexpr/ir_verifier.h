#pragma once

#include <smith/csrc/jit/tensorexpr/fwd_decls.h>
#include <smith/csrc/jit/tensorexpr/ir_visitor.h>

namespace smith::jit::tensorexpr {

class Expr;
class ExprHandle;
class Mod;
class And;
class Or;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;
class Ramp;
class Load;
class IfThenElse;
class Intrinsics;

class Stmt;
class ExternalCall;
class Store;
class For;
class Block;

class SMITH_API IRVerifier : public IRVisitor {
 public:
  IRVerifier() = default;

  void visit(const ModPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;

  void visit(const ExternalCallPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const BlockPtr& v) override;
};

SMITH_API void verify(const StmtPtr& /*s*/);
SMITH_API void verify(const ExprPtr& /*e*/);
SMITH_API void verify(const ExprHandle& /*e*/);

} // namespace smith::jit::tensorexpr
