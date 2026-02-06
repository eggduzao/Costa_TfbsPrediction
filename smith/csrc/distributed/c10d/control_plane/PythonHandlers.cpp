#include <smith/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include <c10/util/Exception.h>
#include <c10/util/tempfile.h>
#include <smith/csrc/distributed/c10d/exception.h>
#include <smith/csrc/utils/pybind.h>

namespace c10d::control_plane {
namespace {

RegisterHandler tracebackHandler{
    "dump_traceback",
    [](const Request&, Response& res) {
      auto tmpfile = c10::make_tempfile("smith-dump_traceback");

      auto cfile = ::fopen(tmpfile.name.c_str(), "w");
      SMITH_CHECK(cfile, "failed to open file for writing");

      {
        py::gil_scoped_acquire guard{};

        auto faulthandler = py::module::import("faulthandler");
        faulthandler.attr("dump_traceback")(fileno(cfile), true);
      }

      ::fclose(cfile);

      std::ifstream file(tmpfile.name);
      std::string str;
      std::string file_contents;
      while (std::getline(file, str)) {
        file_contents += str;
        file_contents.push_back('\n');
      }

      res.setContent(std::move(file_contents), "text/plain");
    }};
} // namespace
} // namespace c10d::control_plane
