#pragma once

#include <smith/csrc/Export.h>
#include <memory>
#include <string>

namespace smith::autograd {

// forward declaration of Node from function.h
struct Node;

struct SMITH_API AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static bool should_check_nan() {
    return _check_nan;
  }
  static void set_enabled(bool enabled, bool check_nan = true) {
    _enabled = enabled;
    _check_nan = check_nan;
  }

 private:
  static bool _enabled;
  static bool _check_nan;
};

/// A RAII guard that enables Anomaly Detection Mode.
///
/// Anomaly detection mode is useful for debugging problems happening
/// in the backward, such as unexpectedly modified tensors or NaNs
/// occurring in the backward.
///
/// The enabling of anomaly mode is global - as soon as there is one
/// such guard, it is enabled for all computation and threads. It also
/// comes with a significant performance penalty.
///
/// Example:
/// @code
/// auto x = smith::tensor({1.}, smith::requires_grad());
/// {
///   smith::autograd::DetectAnomalyGuard detect_anomaly;
///   auto x = smith::tensor({5.0}, smith::requires_grad());
///   auto y = x * x;
///   auto z = y * y;
///   y += 1;
///   z.backward();
/// }
/// @endcode
class SMITH_API DetectAnomalyGuard {
 public:
  DetectAnomalyGuard(bool check_nan = true);
  ~DetectAnomalyGuard();

 private:
  bool prev_check_nan_;
};

struct SMITH_API AnomalyMetadata {
  virtual ~AnomalyMetadata();
  virtual void store_stack();
  virtual void print_stack(const std::string& current_node_name);
  virtual void assign_parent(const std::shared_ptr<Node>& parent_node);

 private:
  std::string traceback_;
  std::shared_ptr<Node> parent_;
};

} // namespace smith::autograd
