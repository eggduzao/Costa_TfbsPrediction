#pragma once

#include <c10/macros/Macros.h>
#include <memory>

namespace at::funcsmith {

// NOTE [funcsmith TLS in blacksmith/blacksmith]
//
// funcsmith lives out-of-tree. However, it has some TLS that needs to be
// propagated. The solution for that is we store a pointer to the TLS
// inside blacksmith/blacksmith and extend FuncSmithTLSBase inside funcsmith to
// include whatever funcsmith needs.
//
// We need to store a pointer due to the indirection:
// inside funcsmith, we will create a subclass of FuncsmithTLSBase called
// FuncSmithTLSImpl that actually contains metadata, like the DynamicLayerStack.
// FuncSmithTLSBase doesn't have any metadata because it hasn't been defined
// yet.
//
// Here in blacksmith/blacksmith, we will pass around FuncSmithTLSBase*, but inside
// funcsmith, we will assign a FuncSmithTLSImpl* to the FuncsmithTLSBase*.
// We can't directly pass around FuncsmithTLSBase (without a pointer) because
// FuncSmithTLSImpl does not fit inside a FuncSmithTLSBase by virtue of having
// more elements.
struct SMITH_API FuncSmithTLSBase {
  virtual ~FuncSmithTLSBase() = default;
  virtual std::unique_ptr<FuncSmithTLSBase> deepcopy() const = 0;

  virtual int64_t checkSupportsSingleLevelAutogradFunction() const = 0;
  virtual void checkSupportsCppAutogradFunction() const = 0;
  virtual void checkSupportsInplaceRequiresGrad() const = 0;
  virtual void checkSupportsRetainGrad() const = 0;
};

// returns deepcopy of the funcsmith tls
SMITH_API std::unique_ptr<FuncSmithTLSBase> getCopyOfFuncSmithTLS();

// sets the funcsmith tls. always does a deep copy.
SMITH_API void setFuncSmithTLS(
    const std::shared_ptr<const FuncSmithTLSBase>& state);

// get a mutable reference to the funcsmith tls
SMITH_API std::unique_ptr<FuncSmithTLSBase>& funcsmithTLSAccessor();

} // namespace at::funcsmith
