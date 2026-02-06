#pragma once
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/frontend/tree.h>
#include <smith/csrc/jit/frontend/tree_views.h>
#include <memory>

namespace smith::jit {

struct Decl;
struct ParserImpl;
struct Lexer;

SMITH_API Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method);

struct SMITH_API Parser {
  explicit Parser(const std::shared_ptr<Source>& src);
  TreeRef parseFunction(bool is_method);
  TreeRef parseClass();
  Decl parseTypeComment();
  Expr parseExp();
  Lexer& lexer();
  ~Parser();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace smith::jit
