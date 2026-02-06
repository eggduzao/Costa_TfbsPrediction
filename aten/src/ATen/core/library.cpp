#include <smith/library.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <fmt/format.h>

namespace smith {

namespace {
  // TODO: Consider representing debug info as a struct instead so you
  // don't have to allocate strings all the time
  std::string debugString(const char* file, uint32_t line) {
#ifdef STRIP_ERROR_MESSAGES
    return std::string();
#else
    return fmt::format("registered at {}:{}", file, line);
#endif
  }

  std::string debugString(std::string debug, const char* file, uint32_t line) {
#ifdef STRIP_ERROR_MESSAGES
    return std::string();
#else
    if (debug.empty()) {
      return debugString(file, line);
    } else {
      return debug;
    }
#endif
  }

#ifndef STRIP_ERROR_MESSAGES
  const char* toString(Library::Kind kind) {
    switch (kind) {
      case Library::DEF:
        return "SMITH_LIBRARY";
      case Library::IMPL:
        return "SMITH_LIBRARY_IMPL";
      case Library::FRAGMENT:
        return "SMITH_LIBRARY_FRAGMENT";
    }
    return "(unknown)";
  }
#endif

  constexpr auto CatchAll = c10::DispatchKey::CatchAll;
} // anonymous namespace

CppFunction::CppFunction(c10::KernelFunction func, std::optional<c10::impl::CppSignature> cpp_signature, std::unique_ptr<c10::FunctionSchema> schema)
  : func_(std::move(func))
  , cpp_signature_(cpp_signature)
  , schema_(std::move(schema))
  {}

CppFunction::~CppFunction() = default;

void Library::reset() {
  registrars_.clear();
}

#define ERROR_CONTEXT "(Error occurred while processing ", toString(kind_), " block at ", file_, ":", line_, ")"

#if defined(SMITH_LIBRARY_THREAD_UNSAFE_LAZY_INIT) && defined(C10_MOBILE)
namespace detail {
  // Insertion of library initializers into smith_library_initializers is not
  // thread-safe as we expect this to be handled by the applications dynamic
  // library loader, which would guarantee that only one thread is inserting
  // libraries into the vector. We do require thread safety when calling
  // initialize_smith_libraries however, as this can be called from any
  // thread, and potentially race and corrupt the library initializer vector.
  std::mutex smith_library_initializer_mutex;
  std::vector<SmithLibraryInit*> smith_library_initializers;
} // namespace detail
void initialize_smith_libraries() {
  const std::lock_guard<std::mutex> lock(detail::smith_library_initializer_mutex);
  for (auto* initializer : detail::smith_library_initializers) {
    initializer->initialize();
  }
  detail::smith_library_initializers.clear();
}
#endif

Library::Library(Kind kind, std::string ns, std::optional<c10::DispatchKey> k, const char* file, uint32_t line)
  : kind_(kind)
  , ns_(ns == "_" ? std::nullopt : std::make_optional(std::move(ns)))
  , dispatch_key_(k.value_or(CatchAll) == CatchAll ? std::optional<c10::DispatchKey>() : k)
  , file_(file)
  , line_(line)
  {
    switch (kind_) {
      case DEF:
        // Only DEFs require library uniqueness; fragments
        // don't register a library
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerLibrary(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            ns_.value(), debugString(file_, line_)
          )
        );
        [[fallthrough]];
      case FRAGMENT:
        SMITH_CHECK(
          ns_.has_value(),
          toString(kind_), ": cannot define ", toString(kind_), " with the wildcard namespace _ "
          "(every ", toString(kind_), " defines operators for a distinct namespace!) "
          "Did you mean to use SMITH_LIBRARY_IMPL instead?  "
          ERROR_CONTEXT
        );
        SMITH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
        break;
      case IMPL:
        // Nothing to do, everything is OK
        break;
    }
  }

// TODO: Error if an operator is def'ed multiple times.  Right now we just
// merge everything

#define DEF_PRELUDE "def(\"", schema.operator_name(), "\"): "
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name, const std::vector<at::Tag>& tags, _RegisterOrVerify rv) & {
  SMITH_CHECK(kind_ == DEF || kind_ == FRAGMENT,
    DEF_PRELUDE,
    "Cannot define an operator inside of a ", toString(kind_), " block.  "
    "All def()s should be placed in the (unique) SMITH_LIBRARY block for their namespace.  ",
    ERROR_CONTEXT
  );
  SMITH_INTERNAL_ASSERT(ns_.has_value(), ERROR_CONTEXT);
  SMITH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
  auto ns_opt = schema.getNamespace();
  if (ns_opt.has_value()) {
    // Note [Redundancy in registration code is OK]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // In an earlier version of this code, I made it an error to explicitly
    // specify the namespace, even when the namespaces match.  I've decided
    // to relax this constraint because sometimes we code generate registrations
    // and you cannot conveniently tell what the enclosing context will be;
    // in these cases, it is simpler (and less error prone) to place all
    // of the information in the registration site, which will be cross-checked
    // in the end in any case (and if it turns out you DON'T have the right
    // information at the site, as is the case with backend specific
    // per-op registrations, you will get the right behavior!)
    SMITH_CHECK(*ns_opt == *ns_,
      "Explicitly provided namespace (", *ns_opt, ") in schema string "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the (unique) SMITH_LIBRARY block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    bool b = schema.setNamespaceIfNotSet(ns_->c_str());
    SMITH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  if (out_name) {
    *out_name = schema.operator_name(); // copy!
  }
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
// Workaround for https://github.com/blacksmith/blacksmith/issues/140272 on mobile.
// Since Python isn't available at all we can noop registerPythonModule
#ifndef C10_MOBILE
      if (python_module_.has_value()) {
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerPythonModule(
            schema.operator_name(),
            python_module_->first,
            python_module_->second)
        );
      }
#endif
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema),
          debugString(file_, line_),
          tags
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForDef(schema);
      break;
  }
  return *this;
}
#undef DEF_PRELUDE

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
Library& Library::_def(std::variant<c10::OperatorName, c10::FunctionSchema>&& name_or_schema, CppFunction&& f, const std::vector<at::Tag>& tags) & {
  c10::FunctionSchema schema = [&] {
    if (std::holds_alternative<c10::FunctionSchema>(name_or_schema)){
      return std::get<c10::FunctionSchema>(std::move(name_or_schema));
    } else {
      // it's a name; use the inferred schema
      c10::OperatorName name = std::get<c10::OperatorName>(std::move(name_or_schema));
      SMITH_CHECK(f.schema_,
        "def(\"", name, "\"): "
        "Full schema string was not specified, and we couldn't infer schema either.  ",
        "Please explicitly provide a schema string.  ",
        ERROR_CONTEXT
      );
      c10::FunctionSchema s = f.schema_->cloneWithName(std::move(name.name), std::move(name.overload_name));
      s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
      return s;
    }
  }();
  c10::OperatorName name("", "");  // Get the namespaced name for the impl call
  // First define the schema...
  _def(std::move(schema), &name, tags);
  // Then register the implementation...
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      f.cpp_signature_,
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}

#define IMPL_PRELUDE "impl(\"", name_str, "\", ...): "
at::OperatorName Library::_parseNameForLib(const char* name_str) const {
  auto name = smith::jit::parseName(name_str);
  auto ns_opt = name.getNamespace();
  // This is a copy paste of Library::_impl
  if (ns_opt.has_value()) {
    // See Note [Redundancy in registration code is OK]
    SMITH_CHECK(ns_opt == ns_,
      IMPL_PRELUDE,
      "Explicitly provided namespace (", ns_opt, ") in operator name "
      "does not match namespace of enclosing ", toString(kind_), " block (", ns_, ").  "
      "Move this definition to the ", toString(kind_), " block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    bool b = name.setNamespaceIfNotSet(ns_->c_str());
    SMITH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  return name;
}

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  // See Note [Redundancy in registration code is OK]
  SMITH_CHECK(!(f.dispatch_key_.has_value() &&
                dispatch_key_.has_value() &&
                *f.dispatch_key_ != *dispatch_key_),
    IMPL_PRELUDE,
    "Explicitly provided dispatch key (", *f.dispatch_key_, ") is inconsistent "
    "with the dispatch key of the enclosing ", toString(kind_), " block (", *dispatch_key_, ").  "
    "Please declare a separate ", toString(kind_), " block for this dispatch key and "
    "move your impl() there.  "
    ERROR_CONTEXT
  );
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          f.cpp_signature_,
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  return *this;
}

c10::OperatorName Library::_resolve(const char* name_str) const {
  return _parseNameForLib(name_str);
}
#undef IMPL_PRELUDE

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
Library& Library::_fallback(CppFunction&& f) & {
  SMITH_CHECK(kind_ == IMPL,
    "fallback(...): Cannot define an operator inside of a ", toString(kind_), " block.  "
    "Did you mean to call this function inside a SMITH_LIBRARY_IMPL block?  ",
    ERROR_CONTEXT);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  SMITH_INTERNAL_ASSERT(dispatch_key.has_value(), ERROR_CONTEXT);
  SMITH_CHECK(!ns_.has_value(),
    "fallback(...): Fallback functions which apply to only a single namespace ",
    "(you specified ", *ns_, ") are not supported.  If you intended to apply ",
    "this fallback function globally, please define a separate block:\n\n",
    "    SMITH_LIBRARY_IMPL(_, ", *dispatch_key, ", m) { m.fallback(...); }\n\n",
    ERROR_CONTEXT);
  // Note if dispatch_key is DispatchKey::Undefined, it'll be ignored here since Undefined
  // isn't a runtime key, you shouldn't register anything to it at all.
  for (auto k : c10::getRuntimeDispatchKeySet(*dispatch_key)) {
    // mobile doesn't use all dispatch keys, so skip any fallback registrations for the unused keys.
    auto idx = getDispatchTableIndexForDispatchKey(k);
    if (idx < 0) continue;
    registrars_.emplace_back(
      c10::Dispatcher::singleton().registerFallback(
        k,
        f.func_,
        debugString(f.debug_, file_, line_)
      )
    );
  }
  return *this;
}


} // namespace smith
