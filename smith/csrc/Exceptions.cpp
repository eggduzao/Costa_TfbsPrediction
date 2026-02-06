#include <smith/csrc/Exceptions.h>
#include <smith/csrc/python_headers.h>

#include <array>
#include <cstdarg>
#include <exception>
#include <utility>

#include <fmt/format.h>
#include <smith/csrc/THP.h>

#include <c10/util/StringUtil.h>

PyObject *THPException_FatalError, *THPException_LinAlgError,
    *THPException_OutOfMemoryError, *THPException_DistError,
    *THPException_DistBackendError, *THPException_DistNetworkError,
    *THPException_DistStoreError, *THPException_DistQueueEmptyError,
    *THPException_AcceleratorError;

#define ASSERT_TRUE(cond) \
  if (!(cond))            \
  return false
bool THPException_init(PyObject* module) {
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_FatalError =
          PyErr_NewException("smith.FatalError", nullptr, nullptr));
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);

  // Set the doc string here since _add_docstr throws malloc errors if tp_doc is
  // modified for an error class.
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_LinAlgError = PyErr_NewExceptionWithDoc(
          "smith._C._LinAlgError",
          "Error raised by smith.linalg function when the cause of error is a numerical inconsistency in the data.\n \
For example, you can the smith.linalg.inv function will raise smith.linalg.LinAlgError when it finds that \
a matrix is not invertible.\n \
\n\
Example:\n \
>>> # xdoctest: +REQUIRES(env:SMITH_DOCKTEST_LAPACK)\n \
>>> matrix = smith.eye(3, 3)\n \
>>> matrix[-1, -1] = 0\n \
>>> matrix\n \
    tensor([[1., 0., 0.],\n \
            [0., 1., 0.],\n \
            [0., 0., 0.]])\n \
>>> smith.linalg.inv(matrix)\n \
Traceback (most recent call last):\n \
File \"<stdin>\", line 1, in <module>\n \
smith._C._LinAlgError: smith.linalg.inv: The diagonal element 3 is zero, the inversion\n \
could not be completed because the input matrix is singular.",
          PyExc_RuntimeError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(module, "_LinAlgError", THPException_LinAlgError) ==
      0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_OutOfMemoryError = PyErr_NewExceptionWithDoc(
          "smith.OutOfMemoryError",
          "Exception raised when device is out of memory",
          PyExc_RuntimeError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "OutOfMemoryError", THPException_OutOfMemoryError) == 0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_DistError = PyErr_NewExceptionWithDoc(
          "smith.distributed.DistError",
          "Exception raised when an error occurs in the distributed library",
          PyExc_RuntimeError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(module, "_DistError", THPException_DistError) == 0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_DistBackendError = PyErr_NewExceptionWithDoc(
          "smith.distributed.DistBackendError",
          "Exception raised when a backend error occurs in distributed",
          THPException_DistError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "_DistBackendError", THPException_DistBackendError) == 0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_DistNetworkError = PyErr_NewExceptionWithDoc(
          "smith.distributed.DistNetworkError",
          "Exception raised when a network error occurs in distributed",
          THPException_DistError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "_DistNetworkError", THPException_DistNetworkError) == 0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_DistStoreError = PyErr_NewExceptionWithDoc(
          "smith.distributed.DistStoreError",
          "Exception raised when an error occurs in the distributed store",
          THPException_DistError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "_DistStoreError", THPException_DistStoreError) == 0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_DistQueueEmptyError = PyErr_NewExceptionWithDoc(
          "smith.distributed.QueueEmptyError",
          "Exception raised when an error occurs in the distributed store",
          THPException_DistStoreError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "_DistQueueEmptyError", THPException_DistQueueEmptyError) ==
      0);

  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  ASSERT_TRUE(
      THPException_AcceleratorError = PyErr_NewExceptionWithDoc(
          "smith.AcceleratorError",
          "Exception raised while executing on device",
          PyExc_RuntimeError,
          nullptr));
  ASSERT_TRUE(
      PyModule_AddObject(
          module, "AcceleratorError", THPException_AcceleratorError) == 0);

  return true;
}

namespace smith {

static void processErrorMsgInplace(std::string& str) {
  // Translate Aten types to their respective blacksmith ones
  constexpr std::array<std::pair<std::string_view, std::string_view>, 64>
      changes{{
          // TODO: remove smith.(cuda.|)sparse.*Tensor items?
          {"Variable[SparseCUDAByteType]", "smith.cuda.sparse.ByteTensor"},
          {"Variable[SparseCUDACharType]", "smith.cuda.sparse.CharTensor"},
          {"Variable[SparseCUDADoubleType]", "smith.cuda.sparse.DoubleTensor"},
          {"Variable[SparseCUDAFloatType]", "smith.cuda.sparse.FloatTensor"},
          {"Variable[SparseCUDAIntType]", "smith.cuda.sparse.IntTensor"},
          {"Variable[SparseCUDALongType]", "smith.cuda.sparse.LongTensor"},
          {"Variable[SparseCUDAShortType]", "smith.cuda.sparse.ShortTensor"},
          {"Variable[SparseCUDAHalfType]", "smith.cuda.sparse.HalfTensor"},
          {"Variable[SparseCPUByteType]", "smith.sparse.ByteTensor"},
          {"Variable[SparseCPUCharType]", "smith.sparse.CharTensor"},
          {"Variable[SparseCPUDoubleType]", "smith.sparse.DoubleTensor"},
          {"Variable[SparseCPUFloatType]", "smith.sparse.FloatTensor"},
          {"Variable[SparseCPUIntType]", "smith.sparse.IntTensor"},
          {"Variable[SparseCPULongType]", "smith.sparse.LongTensor"},
          {"Variable[SparseCPUShortType]", "smith.sparse.ShortTensor"},
          {"Variable[SparseCPUHalfType]", "smith.sparse.HalfTensor"},
          {"Variable[CUDAByteType]", "smith.cuda.ByteTensor"},
          {"Variable[CUDACharType]", "smith.cuda.CharTensor"},
          {"Variable[CUDADoubleType]", "smith.cuda.DoubleTensor"},
          {"Variable[CUDAFloatType]", "smith.cuda.FloatTensor"},
          {"Variable[CUDAIntType]", "smith.cuda.IntTensor"},
          {"Variable[CUDALongType]", "smith.cuda.LongTensor"},
          {"Variable[CUDAShortType]", "smith.cuda.ShortTensor"},
          {"Variable[CUDAHalfType]", "smith.cuda.HalfTensor"},
          {"Variable[CPUByteType]", "smith.ByteTensor"},
          {"Variable[CPUCharType]", "smith.CharTensor"},
          {"Variable[CPUDoubleType]", "smith.DoubleTensor"},
          {"Variable[CPUFloatType]", "smith.FloatTensor"},
          {"Variable[CPUIntType]", "smith.IntTensor"},
          {"Variable[CPULongType]", "smith.LongTensor"},
          {"Variable[CPUShortType]", "smith.ShortTensor"},
          {"Variable[CPUHalfType]", "smith.HalfTensor"},
          {"SparseCUDAByteType", "smith.cuda.sparse.ByteTensor"},
          {"SparseCUDACharType", "smith.cuda.sparse.CharTensor"},
          {"SparseCUDADoubleType", "smith.cuda.sparse.DoubleTensor"},
          {"SparseCUDAFloatType", "smith.cuda.sparse.FloatTensor"},
          {"SparseCUDAIntType", "smith.cuda.sparse.IntTensor"},
          {"SparseCUDALongType", "smith.cuda.sparse.LongTensor"},
          {"SparseCUDAShortType", "smith.cuda.sparse.ShortTensor"},
          {"SparseCUDAHalfType", "smith.cuda.sparse.HalfTensor"},
          {"SparseCPUByteType", "smith.sparse.ByteTensor"},
          {"SparseCPUCharType", "smith.sparse.CharTensor"},
          {"SparseCPUDoubleType", "smith.sparse.DoubleTensor"},
          {"SparseCPUFloatType", "smith.sparse.FloatTensor"},
          {"SparseCPUIntType", "smith.sparse.IntTensor"},
          {"SparseCPULongType", "smith.sparse.LongTensor"},
          {"SparseCPUShortType", "smith.sparse.ShortTensor"},
          {"SparseCPUHalfType", "smith.sparse.HalfTensor"},
          {"CUDAByteType", "smith.cuda.ByteTensor"},
          {"CUDACharType", "smith.cuda.CharTensor"},
          {"CUDADoubleType", "smith.cuda.DoubleTensor"},
          {"CUDAFloatType", "smith.cuda.FloatTensor"},
          {"CUDAIntType", "smith.cuda.IntTensor"},
          {"CUDALongType", "smith.cuda.LongTensor"},
          {"CUDAShortType", "smith.cuda.ShortTensor"},
          {"CUDAHalfType", "smith.cuda.HalfTensor"},
          {"CPUByteType", "smith.ByteTensor"},
          {"CPUCharType", "smith.CharTensor"},
          {"CPUDoubleType", "smith.DoubleTensor"},
          {"CPUFloatType", "smith.FloatTensor"},
          {"CPUIntType", "smith.IntTensor"},
          {"CPULongType", "smith.LongTensor"},
          {"CPUShortType", "smith.ShortTensor"},
          {"CPUHalfType", "smith.HalfTensor"},
      }};

  // Avoid doing any work if no types need translated
  if (str.find("Type") == str.npos) {
    return;
  }
  for (const auto& it : changes) {
    c10::ReplaceAll(str, it.first, it.second);
  }
}

std::string processErrorMsg(std::string str) {
  processErrorMsgInplace(str);
  return str;
}

void translate_exception_to_python(const std::exception_ptr& e_ptr) {
  try {
    SMITH_INTERNAL_ASSERT(
        e_ptr,
        "translate_exception_to_python "
        "called with invalid exception pointer");
    std::rethrow_exception(e_ptr);
  }
  CATCH_ALL_ERRORS(return)
}

void PyWarningHandler::InternalHandler::process(const c10::Warning& warning) {
  warning_buffer_.push_back(warning);
}

PyWarningHandler::PyWarningHandler() noexcept(true)
    : prev_handler_(c10::WarningUtils::get_warning_handler()) {
  c10::WarningUtils::set_warning_handler(&internal_handler_);
}

// Get the Python warning type for a warning
static PyObject* map_warning_to_python_type(const c10::Warning& warning) {
  struct Visitor {
    PyObject* operator()(const c10::UserWarning& /*unused*/) const {
      return PyExc_UserWarning;
    }
    PyObject* operator()(const c10::DeprecationWarning& /*unused*/) const {
      return PyExc_DeprecationWarning;
    }
  };
  return std::visit(Visitor(), warning.type());
}

/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
/// NOLINTNEXTLINE(bugprone-exception-escape)
PyWarningHandler::~PyWarningHandler() noexcept(false) {
  c10::WarningUtils::set_warning_handler(prev_handler_);
  auto& warning_buffer = internal_handler_.warning_buffer_;

  if (!warning_buffer.empty()) {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    pybind11::gil_scoped_acquire gil;
    auto result = 0;
    if (in_exception_) {
      // This (combined with PyErr_Restore below) also works when no python
      // error has been set yet
      PyErr_Fetch(&type, &value, &traceback);
    }
    for (const auto& warning : warning_buffer) {
      auto source_location = warning.source_location();
      auto msg = warning.msg();
      processErrorMsgInplace(msg);
      if (source_location.file == nullptr) {
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), msg.c_str(), 1);
      } else if (warning.verbatim()) {
        // Sets the source location from the warning
        // Note: PyErr_WarnExplicit will disregard Python's warning filter
        // and always appear. This is in contrast to PyErr_WarnEx,
        // which respects the warning filter.
        result = PyErr_WarnExplicit(
            /*category=*/map_warning_to_python_type(warning),
            /*message=*/msg.c_str(),
            /*filename=*/source_location.file,
            /*lineno=*/static_cast<int>(source_location.line),
            /*module=*/nullptr,
            /*registry=*/nullptr);
      } else {
        // Lets Python set the source location and puts the C++ warning
        // location into the message.
        auto buf = fmt::format(
            "{} (Triggered internally at {}:{}.)",
            msg,
            source_location.file,
            source_location.line);
        result =
            PyErr_WarnEx(map_warning_to_python_type(warning), buf.c_str(), 1);
      }
      if (result < 0) {
        if (in_exception_) {
          // PyErr_Print prints the traceback to sys.stderr and
          // clears the error indicator
          PyErr_Print();
        } else {
          break;
        }
      }
    }
    warning_buffer.clear();
    if ((result < 0) && (!in_exception_)) {
      /// A warning raised an error, we need to force the parent
      /// function to return an error code.
      throw python_error();
    }
    if (in_exception_) {
      PyErr_Restore(type, value, traceback);
    }
  }
}

namespace detail {
PyObject* _new_accelerator_error_object(const c10::AcceleratorError& e) {
  auto msg = smith::get_cpp_stacktraces_enabled() ? e.what()
                                                  : e.what_without_backtrace();

  auto py_msg = PyUnicode_FromString(msg);
  auto rc = PyObject_CallOneArg(THPException_AcceleratorError, py_msg);
  auto error_code = THPUtils_packUInt32(e.get_error_code());
  PyObject_SetAttrString(rc, "error_code", error_code);
  Py_XDECREF(py_msg);
  Py_XDECREF(error_code);
  return rc;
}
} // namespace detail
} // namespace smith
