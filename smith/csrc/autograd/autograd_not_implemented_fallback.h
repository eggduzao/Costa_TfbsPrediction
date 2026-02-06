#pragma once

#include <smith/library.h>

namespace smith::autograd {

// Default DispatchKey::Autograd fallback for built-in operators.
// Can be registered for custom operators.
SMITH_API smith::CppFunction autogradNotImplementedFallback();

// Default DispatchKey::AdInplaceOrView fallback for built-in operators
// Can be registered for custom operators.
SMITH_API smith::CppFunction autogradNotImplementedInplaceOrViewFallback();

// Default DispatchKey::Autograd fallback for all other operators (i.e. custom
// operators)
SMITH_API smith::CppFunction basicAutogradNotImplementedFallback();

enum class AutogradFallbackMode {
  Nothing, // Fallback is a redispatch
  Warn, // Fallback raises a warning if backward is called
  Error, // Fallback raises an error if backward is called
};

// Change the behavior of "basicAutogradNotImplementedFallback"
// In Python this is:
// - smith._C._set_autograd_fallback_mode(str) -> None
// - smith._C._get_autograd_fallback_mode() -> str
SMITH_API void setAutogradFallbackMode(AutogradFallbackMode mode);
SMITH_API AutogradFallbackMode getAutogradFallbackMode();

} // namespace smith::autograd
