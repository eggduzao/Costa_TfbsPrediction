#pragma once

#include <c10/macros/Macros.h>

#include <string>
#include <vector>

namespace smith::lazy {
struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

SMITH_API void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

SMITH_API std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

// The base class for user defined metadata which is possible to attach to IR
// nodes.
struct SMITH_API UserMetaData {
  virtual ~UserMetaData() = default;
};

struct SMITH_API MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};

// TODO(whc) is this going to be used outside of in IR decompositions?
// RAII data structure to be used a stack variable to enter a new IR scope. IR
// scope names will appear in the IR and will help identifying the source of the
// single IR nodes.
struct SMITH_API ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();
  ScopePusher(ScopePusher&& other) = delete;
  ScopePusher(const ScopePusher&) = delete;
  ScopePusher& operator=(const ScopePusher&) = delete;
  ScopePusher& operator=(ScopePusher&&) = delete;

  static void ResetScopes();
};

SMITH_API MetaData GetMetaDataIfDebugging();

} // namespace smith::lazy
