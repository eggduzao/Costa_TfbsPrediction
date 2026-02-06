#include <smith/csrc/tensor/python_tensor.h>

#include <pybind11/pybind11.h>
#include <structmember.h>
#include <smith/csrc/utils/pybind.h>

#include <smith/csrc/Dtype.h>
#include <smith/csrc/DynamicTypes.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/Layout.h>
#include <smith/csrc/autograd/generated/VariableType.h>
#include <smith/csrc/autograd/python_variable.h>
#include <smith/csrc/autograd/utils/wrap_outputs.h>
#include <smith/csrc/autograd/variable.h>
#include <smith/csrc/utils/cuda_enabled.h>
#include <smith/csrc/utils/device_lazy_init.h>
#include <smith/csrc/utils/python_strings.h>
#include <smith/csrc/utils/tensor_new.h>
#include <smith/csrc/utils/tensor_types.h>

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace smith::tensors {

using namespace at;
using namespace smith::autograd;

struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  bool is_xpu;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  char name[64];
  int backend;
  int scalar_type;

  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(static_cast<Backend>(backend));
  }

  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

static_assert(
    std::is_standard_layout_v<PyTensorType>,
    "PyTensorType must be standard layout");

static Backend default_backend = Backend::CPU;

static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types);

static PyObject* Tensor_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  SMITH_CHECK_TYPE(
      !tensor_type.is_cuda || smith::utils::cuda_enabled(),
      "type ",
      tensor_type.name,
      " not available. Smith not compiled with CUDA enabled.")
  if (tensor_type.is_cuda) {
    SMITH_WARN_ONCE(
        "The smith.cuda.*DtypeTensor constructors are no longer recommended. "
        "It's best to use methods such as smith.tensor(data, dtype=*, device='cuda') to create tensors.")
  }
  return THPVariable_Wrap(smith::utils::legacy_tensor_ctor(
      tensor_type.get_dispatch_key(),
      tensor_type.get_scalar_type(),
      args,
      kwargs));
  END_HANDLE_TH_ERRORS
}

// TODO: Deprecate this instancecheck entirely.  It's here to make
// instanceof(t, smith.FloatTensor) work, but we are not going to keep
// adding smith.QuantizedIntTensor classes for every new tensor type
// we add...
static PyObject* Tensor_instancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);
    // NB: This is a little unfortunate, in that if I do an isinstance check
    // against smith.cuda.FloatTensor, this will immediately initialize CUDA.
    // I originally thought that it would not be possible for aten_type_ to
    // be nullptr if you had a tensor of some type, in which case you can
    // skip initializing aten_type(), but TestAutograd.test_type_conversions
    // seems to violate this property (for whatever reason.)
    //
    // TODO: Stop using legacyExtractDispatchKey here (probably need to build
    // in instanceof checking to Tensor class itself)
    if (legacyExtractDispatchKey(var.key_set()) == self->get_dispatch_key() &&
        var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_dtype(PyTensorType* self, void* unused) {
  return smith::autograd::utils::wrap(self->dtype);
}

static PyObject* Tensor_layout(PyTensorType* self, void* unused) {
  return smith::autograd::utils::wrap(self->layout);
}

static PyObject* Tensor_is_cuda(PyTensorType* self, void* unused) {
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* Tensor_is_xpu(PyTensorType* self, void* unused) {
  if (self->is_xpu) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* Tensor_is_sparse(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

static PyObject* Tensor_is_sparse_csr(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::SparseCsr) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyMethodDef metaclass_methods[] = {
    {"__instancecheck__", Tensor_instancecheck, METH_O, nullptr},
    {nullptr}};

typedef PyObject* (*getter)(PyObject*, void*);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef metaclass_properties[] = {
    {"dtype", (getter)Tensor_dtype, nullptr, nullptr, nullptr},
    {"layout", (getter)Tensor_layout, nullptr, nullptr, nullptr},
    {"is_cuda", (getter)Tensor_is_cuda, nullptr, nullptr, nullptr},
    {"is_xpu", (getter)Tensor_is_xpu, nullptr, nullptr, nullptr},
    {"is_sparse", (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
    {"is_sparse_csr", (getter)Tensor_is_sparse_csr, nullptr, nullptr, nullptr},
    {nullptr}};

static PyTypeObject metaclass = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "smith.tensortype", /* tp_name */
    sizeof(PyTypeObject) /* tp_basicsize */
};

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  // NOLINTNEXTLINE(misc-redundant-expression)
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static PyTypeObject tensor_type_prototype = {
    PyVarObject_HEAD_INIT(&metaclass, 0)
    nullptr, /* tp_name */
    sizeof(PyTensorType) /* tp_basicsize */
};

static void py_initialize_tensor_type(
    PyTypeObject& type,
    const char* name,
    PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // We copy the basic object fields from a prototype definition and initialize
  // the remaining fields below.
  memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
  // Subclassing from smith.<ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing smith.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = Tensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static std::string get_name(Backend backend, ScalarType scalarType) {
  std::ostringstream ss;
  ss << smith::utils::backend_to_string(backend) << '.' << toString(scalarType)
     << "Tensor";
  return ss.str();
}

static THPObjectPtr get_storage_obj(Backend backend, ScalarType dtype) {
  auto module_name = smith::utils::backend_to_string(backend);
  auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name));
  if (!module_obj)
    throw python_error();

  auto storage_name = std::string(toString(dtype)) + "Storage";
  THPObjectPtr storage(
      PyObject_GetAttrString(module_obj.get(), storage_name.c_str()));
  SMITH_CHECK_TYPE(
      storage.get(), "couldn't find storage object ", storage_name);
  return storage;
}

static void set_type(
    PyTensorType& type_obj,
    Backend backend,
    ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout =
      (THPLayout*)Py_NewRef(smith::getTHPLayout(layout_from_backend(backend)));
  type_obj.dtype = (THPDtype*)Py_NewRef(smith::getTHPDtype(scalarType));
  type_obj.is_cuda =
      (backend == at::Backend::CUDA || backend == at::Backend::SparseCUDA);
  type_obj.is_xpu =
      (backend == at::Backend::XPU || backend == at::Backend::SparseXPU);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto smith = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith)
    throw python_error();

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(smith, "Tensor"));
  if (!tensor_class)
    throw python_error();

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  SMITH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res)
    throw python_error();

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

// A note about the lifetime of the various PyTensorType: normally
// PyTypeObject instances are statically allocated, but we want to create them
// dynamically at init time, because their exact number depends on
// smith::utils::all_declared_types(). The memory for each PyTensorType is
// allocated by initialize_aten_types() and never freed: technically it's a
// leak, but it's not a problem since we want them to be alive for the whole
// time of the process anyway.
//
// An alternative is to use a std::vector<PyTensorType> instead, and let
// std::vector to manage the lifetime of its items. This is problematic
// though, because it means that the memory of PyTensorType is deallocated at
// some point during the exit: if by chance we have another global destructor
// and/or atexit() function which tries to access the PyTensorTypes, we risk
// an use-after-free error. This happens for example if we embed CPython and
// call Py_Finalize inside an atexit() function which was registered before
// importing smith.
static std::vector<PyTensorType*> tensor_types;

static void set_default_storage_type(Backend backend, ScalarType dtype) {
  THPObjectPtr storage = get_storage_obj(backend, dtype);

  auto smith_module = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith_module)
    throw python_error();

  if (PyObject_SetAttrString(smith_module.get(), "Storage", storage) != 0) {
    throw python_error();
  }
}

static void set_default_tensor_type(
    std::optional<Backend> backend,
    std::optional<ScalarType> dtype) {
  if (backend.has_value()) {
    SMITH_CHECK_TYPE(
        *backend != Backend::Undefined, "default type cannot be undefined");
    SMITH_CHECK_TYPE(
        !isSparse(*backend),
        "only dense types are supported as the default type");
  }
  if (dtype.has_value()) {
    SMITH_CHECK_TYPE(
        at::isFloatingType(*dtype),
        "only floating-point types are supported as the default type");
  }

  // Try setting default storage in python first as it's the only operation that
  // can fail
  set_default_storage_type(
      backend.value_or(default_backend),
      dtype.value_or(at::get_default_dtype_as_scalartype()));

  if (dtype.has_value()) {
    at::set_default_dtype(scalarTypeToTypeMeta(*dtype));
  }
  if (backend.has_value()) {
    default_backend = *backend;
  }
}

static void initialize_aten_types(std::vector<PyTensorType*>& tensor_types) {
  // includes CUDA types even when Blacksmith is not built with CUDA
  auto declared_types = smith::utils::all_declared_types();
  tensor_types.resize(declared_types.size());

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    tensor_types[i] = new PyTensorType();
    auto& tensor_type = *tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }

  set_default_tensor_type(Backend::CPU, ScalarType::Float);
}

void initialize_python_bindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the smith.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `smith.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object smith.FloatTensor, smith.DoubleTensor,
  // etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(
        tensor_type->py_type, tensor_type->name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. smith.FloatTensor
  // is added to the `smith` module as `FloatTensor`. Also add all the type
  // objects to the set smith._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types) {
  auto smith_module = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith_module)
    throw python_error();

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(smith_module.get(), "_tensor_classes"));
  if (!tensor_classes)
    throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type->name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj)
      throw python_error();

    PyObject* type_obj = (PyObject*)tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static bool PyTensorType_Check(PyObject* obj) {
  auto it = std::find_if(
      tensor_types.begin(), tensor_types.end(), [obj](PyTensorType* x) {
        return (PyObject*)x == obj;
      });
  return it != tensor_types.end();
}

void py_set_default_tensor_type(PyObject* obj) {
  SMITH_WARN_ONCE(
      "smith.set_default_tensor_type() is deprecated as of Blacksmith 2.1, "
      "please use smith.set_default_dtype() and smith.set_default_device() as alternatives.")
  SMITH_CHECK_TYPE(
      PyTensorType_Check(obj),
      "invalid type object: only floating-point types are supported as the default type");
  PyTensorType* type = (PyTensorType*)obj;
  SMITH_CHECK_TYPE(
      !type->is_cuda || smith::utils::cuda_enabled(),
      "type ",
      type->name,
      " not available. Smith not compiled with CUDA enabled.")
  set_default_tensor_type(type->get_backend(), type->get_scalar_type());
}

void py_set_default_dtype(PyObject* obj) {
  SMITH_CHECK_TYPE(
      THPDtype_Check(obj),
      "invalid dtype object: only floating-point types are supported as the default type");
  auto scalar_type = ((THPDtype*)obj)->scalar_type;
  set_default_tensor_type(/*backend=*/std::nullopt, scalar_type);
}

c10::DispatchKey get_default_dispatch_key() {
  return backendToDispatchKey(default_backend);
}

at::Device get_default_device() {
  return at::Device(c10::backendToDeviceType(default_backend));
}

ScalarType get_default_scalar_type() {
  return get_default_dtype_as_scalartype();
}

} // namespace smith::tensors
