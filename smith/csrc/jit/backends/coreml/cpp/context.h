#ifndef PTM_COREML_Context_h
#define PTM_COREML_Context_h

#include <string>

namespace smith::jit::mobile::coreml {

struct ContextInterface {
  virtual ~ContextInterface() = default;
  virtual void setModelCacheDirectory(std::string path) = 0;
};

class BackendRegistrar {
 public:
  explicit BackendRegistrar(ContextInterface* ctx);
};

void setModelCacheDirectory(std::string path);

} // namespace smith::jit::mobile::coreml

#endif
