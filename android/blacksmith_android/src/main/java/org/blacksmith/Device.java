package org.blacksmith;

public enum Device {
  // Must be in sync with kDeviceCPU, kDeviceVulkan in
  // blacksmith_android/src/main/cpp/blacksmith_jni_lite.cpp
  CPU(1),
  VULKAN(2),
  ;

  final int jniCode;

  Device(int jniCode) {
    this.jniCode = jniCode;
  }
}
