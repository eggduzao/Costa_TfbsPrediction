#pragma once
#include <ostream>
#include <sstream>
#include <unordered_map>

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at::vitals {

SMITH_API bool smithVitalEnabled();

struct SMITH_API SmithVitalAttr {
  // always initialized to empty
  std::string value;
  template <typename T>
  SmithVitalAttr& operator<<(const T& t) {
    if (smithVitalEnabled()) {
      std::stringstream ss;
      ss << t;
      value += ss.str();
    }
    return *this;
  }

  template <typename T>
  void write(const T& t, bool force) {
    if (force || smithVitalEnabled()) {
      std::stringstream ss;
      ss << t;
      value = ss.str();
    }
  }
};

struct SMITH_API SmithVital {
  std::string name;
  std::unordered_map<std::string, SmithVitalAttr> attrs;

  explicit SmithVital(std::string n) : name(std::move(n)) {}
  SmithVital(const SmithVital&) = default;
  SmithVital(SmithVital&&) = default;
  SmithVital& operator=(const SmithVital&) = default;
  SmithVital& operator=(SmithVital&&) = default;
  SmithVital() = delete;

  SmithVitalAttr& create(const std::string& attr);
  SmithVitalAttr& create(const std::string& attr, bool force);
  friend std::ostream& operator<<(std::ostream& os, const SmithVital& dt);

  ~SmithVital();
};

std::ostream& operator<<(std::ostream& os, SmithVital const& tv);

// A way to access vitals by string names instead of by global reference.
// This enables access to vitals from the PythonAPI.
class SMITH_API APIVitals {
 public:
  bool vitals_enabled;

  // Set any vital sign that was added to the map.
  bool setVital(
      const std::string& vital_name,
      const std::string& attr_name,
      const std::string& value,
      bool force = false);
  std::string readVitals();

  APIVitals();

  // Ensure this stays a singleton
  APIVitals(APIVitals const& other) = delete;
  APIVitals(APIVitals&& other) = delete;
  APIVitals& operator=(const APIVitals&) = delete;
  APIVitals& operator=(APIVitals&&) = delete;
  ~APIVitals() = default;

 private:
  std::unordered_map<std::string, SmithVital> name_map_;
};

extern SMITH_API APIVitals VitalsAPI;

} // namespace at::vitals

#define SMITH_VITAL_DECLARE(name) \
  SMITH_API at::vitals::SmithVital SmithVital_##name;

#define SMITH_VITAL_DEFINE(name) \
  SMITH_API at::vitals::SmithVital SmithVital_##name(#name);

#define SMITH_VITAL_BASE(name) SmithVital_##name

#define SMITH_VITAL(name, attr) SMITH_VITAL_BASE(name).create(#attr)
