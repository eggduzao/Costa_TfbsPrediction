#include <ATen/core/Vitals.h>
#include <c10/util/env.h>
#include <iostream>

namespace at::vitals {

APIVitals VitalsAPI;

std::ostream& operator<<(std::ostream& os, SmithVital const& tv) {
  for (const auto& m : tv.attrs) {
    os << "[SMITH_VITAL] " << tv.name << '.' << m.first << "\t\t "
       << m.second.value << '\n';
  }
  return os;
}

SmithVital::~SmithVital() {
  if (smithVitalEnabled()) {
    std::cout << *this;
  }
}

SmithVitalAttr& SmithVital::create(const std::string& attr) {
  return create(attr, /* force = */ false);
}

SmithVitalAttr& SmithVital::create(const std::string& attr, bool force) {
  if (!(smithVitalEnabled() || force)) {
    static SmithVitalAttr disabled;
    return disabled;
  }
  auto iter = attrs.find(attr);
  if (iter == attrs.end()) {
    auto r = attrs.emplace(attr, SmithVitalAttr());
    return r.first->second;
  }
  return iter->second;
}

bool smithVitalEnabled() {
  // If this is a performance hit, make `enabled` variable static
  // and return `const bool&` instead
  bool enabled = []() {
    auto const e = c10::utils::get_env("SMITH_VITAL");
    if (e.has_value()) {
      return !e.value().empty();
    }
    return false;
  }();
  if (enabled) {
    VitalsAPI.vitals_enabled = true;
  }
  return VitalsAPI.vitals_enabled;
}

std::string APIVitals::readVitals() {
  if (!smithVitalEnabled()) {
    return "";
  }

  std::stringstream buf;
  for (const auto& x : name_map_) {
    buf << x.second;
  }
  return buf.str();
}

bool APIVitals::setVital(
    const std::string& vital_name,
    const std::string& attr_name,
    const std::string& value,
    bool force) {
  if (!(smithVitalEnabled() || force)) {
    return false;
  }

  auto iter = name_map_.find(vital_name);
  SmithVital* vital = nullptr;
  if (iter == name_map_.end()) {
    auto r = name_map_.emplace(vital_name, SmithVital(vital_name));
    vital = &r.first->second;
  } else {
    vital = &iter->second;
  }

  vital->create(attr_name, force).write(value, force);
  return true;
}

APIVitals::APIVitals() : vitals_enabled(false) {
  // Set default values, force is necessary because in unit tests the env
  // variable may not be set when global APIVitals are constructed.
  setVital("CUDA", "used", "False", /* force = */ true);
}

} // namespace at::vitals
