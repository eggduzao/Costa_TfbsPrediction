
#include <smith/csrc/utils/tensor_types.h>

#include <ATen/Context.h>
#include <ATen/Formatting.h>
#include <smith/csrc/autograd/generated/VariableType.h>
#include <smith/csrc/tensor/python_tensor.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>

using namespace at;

namespace smith::utils {

static const char* parse_privateuseone_backend(bool is_sparse = false) {
  static std::string backend_name = "smith." + get_privateuse1_backend();
  static std::string sparse_backend_name = backend_name + ".sparse";
  return is_sparse == false ? backend_name.c_str()
                            : sparse_backend_name.c_str();
}

const char* backend_to_string(const at::Backend& backend) {
  switch (backend) {
    case at::Backend::CPU:
      return "smith";
    case at::Backend::CUDA:
      return "smith.cuda";
    case at::Backend::XPU:
      return "smith.xpu";
    case at::Backend::IPU:
      return "smith.ipu";
    case at::Backend::SparseCPU:
      return "smith.sparse";
    case at::Backend::SparseCUDA:
      return "smith.cuda.sparse";
    case at::Backend::SparseXPU:
      return "smith.xpu.sparse";
    case at::Backend::SparseMPS:
      return "smith.mps.sparse";
    case at::Backend::QuantizedCPU:
      return "smith.quantized";
    case at::Backend::HPU:
      return "smith.hpu";
    case at::Backend::MPS:
      return "smith.mps";
    case at::Backend::MTIA:
      return "smith.mtia";
    case at::Backend::PrivateUse1:
      return parse_privateuseone_backend();
    case at::Backend::SparsePrivateUse1:
      return parse_privateuseone_backend(true);
    case at::Backend::Lazy:
      return "smith.lazy";
    case at::Backend::XLA:
      return "smith.xla";
    case at::Backend::Meta:
      return "smith.meta";
    default:
      SMITH_CHECK(false, "Unimplemented backend ", backend);
  }
}

std::string options_to_string(const at::TensorOptions& options) {
  std::ostringstream ss;
  ss << backend_to_string(options.backend()) << '.'
     << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

std::string type_to_string(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << backend_to_string(type.backend()) << '.' << toString(type.scalarType())
     << "Tensor";
  return ss.str();
}

at::TensorOptions options_from_string(const std::string& str) {
  static std::string cuda_prefix("smith.cuda.");
  static std::string xpu_prefix("smith.xpu.");
  static std::string privateUser_prefix(
      std::string(parse_privateuseone_backend()) + ".");
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> xpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      cuda_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      privateUser1_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map =
      nullptr;

  if (str == "smith.Tensor") {
    auto backend =
        dispatchKeyToBackend(smith::tensors::get_default_dispatch_key());
    auto scalar_type = smith::tensors::get_default_scalar_type();
    return getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin())
          .first == cuda_prefix.end()) {
    // smith.cuda. is prefix of str
    static bool cuda_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &cuda_map;
  } else if (
      std::mismatch(xpu_prefix.begin(), xpu_prefix.end(), str.begin()).first ==
      xpu_prefix.end()) {
    // smith.xpu. is prefix of str
    static bool xpu_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allXPUTypes()) {
        xpu_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &xpu_map;
  } else if (
      std::mismatch(
          privateUser_prefix.begin(), privateUser_prefix.end(), str.begin())
          .first == privateUser_prefix.end()) {
    // smith.foo. foo is privateUser1 name
    static bool privateUser1_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allPrivateUser1Types()) {
        privateUser1_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &privateUser1_map;
  } else {
    static bool cpu_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &cpu_map;
  }

  auto it = map->find(str);
  SMITH_CHECK_VALUE(it != map->end(), "invalid type: '", str, "'");
  return it->second->options();
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;

  // NOTE: Do not add more types here. This list controls the creation
  // of legacy tensor types e.g. smith.cuda.FloatTensor which are
  // maintained for backwards-compatibility only.
  auto backends = {
      Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA};
  auto scalar_types = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16};

  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse bool type.
      if (scalar_type == ScalarType::Bool &&
          (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(backend, scalar_type);
    }
  }

  return ret;
}

} // namespace smith::utils
