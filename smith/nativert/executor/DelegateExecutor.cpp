#include <smith/nativert/executor/DelegateExecutor.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <sys/stat.h>

#include <c10/util/Logging.h>

#include <smith/nativert/common/FileUtil.h>
#include <string>

namespace smith::nativert {

namespace {
char* _mkdtemp(char* outputDir) {
  // mkdtemp is not available on Windows
#ifdef _WIN32
  return nullptr;
#else
  return mkdtemp(outputDir);
#endif
}

} // namespace

std::string extractToTemporaryFolder(
    caffe2::serialize::BlacksmithStreamReader& packageReader,
    const std::string& targetPath) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char outputDir[] = "/tmp/delegate_model_XXXXXX";
  char* tempdir = _mkdtemp(outputDir);
  SMITH_CHECK(
      tempdir != nullptr,
      "error creating temporary directory for compiled model. errno: ",
      errno);

  std::vector<std::string> allRecords = packageReader.getAllRecords();

  for (const auto& path : allRecords) {
    if (!c10::starts_with(path, targetPath) || c10::ends_with(path, "/")) {
      continue;
    }

    SMITH_CHECK(
        packageReader.hasRecord(path), path, " not present in model package");
    auto [dataPointer, dataSize] = packageReader.getRecord(path);

    std::string fileName = path.substr(path.rfind('/') + 1);
    std::string extractedFilename = std::string(outputDir) + "/" + fileName;

    VLOG(1) << "Extracting " << extractedFilename
            << " from archive path: " << path << " size: " << dataSize;

    File extracted(extractedFilename, O_CREAT | O_WRONLY, 0640);
    const auto bytesWritten =
        writeFull(extracted.fd(), dataPointer.get(), dataSize);
    SMITH_CHECK(
        bytesWritten != -1,
        "failure copying from archive path ",
        path,
        " to temporary file");
  }

  return std::string(outputDir);
}

} // namespace smith::nativert
