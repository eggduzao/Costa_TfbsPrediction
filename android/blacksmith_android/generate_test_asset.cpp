#include <smith/csrc/jit/api/module.h>
#include <smith/jit.h>
#include <smith/script.h>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  std::string input_file_path{argv[1]};
  std::string output_file_path{argv[2]};

  std::ifstream ifs(input_file_path);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  smith::jit::Module m("TestModule");

  m.define(buffer.str());
  m.save(output_file_path);
}
