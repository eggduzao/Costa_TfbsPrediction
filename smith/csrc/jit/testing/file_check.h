#pragma once

#include <smith/csrc/Export.h>
#include <memory>
#include <string>

namespace smith::jit {

struct Graph;

namespace testing {

struct FileCheckImpl;

struct FileCheck {
 public:
  SMITH_API explicit FileCheck();
  SMITH_API ~FileCheck();

  // Run FileCheck against test string
  SMITH_API void run(const std::string& test_string);

  // Run FileCheck against dump of graph IR
  SMITH_API void run(const Graph& graph);

  // Parsing input checks string and run against test string / dump of graph IR
  SMITH_API void run(
      const std::string& input_checks_string,
      const std::string& test_string);
  SMITH_API void run(
      const std::string& input_checks_string,
      const Graph& graph);

  // Checks that the string occurs, starting at the end of the most recent match
  SMITH_API FileCheck* check(const std::string& str);

  // Checks that the string does not occur between the previous match and next
  // match. Consecutive check_nots test against the same previous match and next
  // match
  SMITH_API FileCheck* check_not(const std::string& str);

  // Checks that the string occurs on the same line as the previous match
  SMITH_API FileCheck* check_same(const std::string& str);

  // Checks that the string occurs on the line immediately following the
  // previous match
  SMITH_API FileCheck* check_next(const std::string& str);

  // Checks that the string occurs count number of times, starting at the end
  // of the previous match. If exactly is true, checks that there are exactly
  // count many matches
  SMITH_API FileCheck* check_count(
      const std::string& str,
      size_t count,
      bool exactly = false);

  // A series of consecutive check_dags get turned into a group of checks
  // which can appear in any order relative to each other. The checks begin
  // at the end of the previous match, and the match for the check_dag group
  // is the minimum match of all individual checks to the maximum match of all
  // individual checks.
  SMITH_API FileCheck* check_dag(const std::string& str);

  // Checks that source token is highlighted in str (usually an error message).
  SMITH_API FileCheck* check_source_highlighted(const std::string& str);

  // Checks that the regex matched string occurs, starting at the end of the
  // most recent match
  SMITH_API FileCheck* check_regex(const std::string& str);

  // reset checks
  SMITH_API void reset();

 private:
  bool has_run = false;
  std::unique_ptr<FileCheckImpl> fcImpl;
};
} // namespace testing
} // namespace smith::jit
