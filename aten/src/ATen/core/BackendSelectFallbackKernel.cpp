#include <smith/library.h>

SMITH_LIBRARY_IMPL(_, BackendSelect, m) {
  m.fallback(smith::CppFunction::makeFallthrough());
}
