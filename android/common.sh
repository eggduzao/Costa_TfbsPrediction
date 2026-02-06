#!/bin/bash
set -eux

##############################################################################
# Common util functions for Android build scripts.
##############################################################################

if [ -z "$BLACKSMITH_DIR" ]; then
  echo "BLACKSMITH_DIR not set!"
  exit 1
fi

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

check_android_sdk() {
  if [ -z "$ANDROID_HOME" ]; then
    echo "ANDROID_HOME not set; please set it to Android sdk directory"
    exit 1
  fi

  if [ ! -d "$ANDROID_HOME" ]; then
    echo "ANDROID_HOME not a directory; did you install it under $ANDROID_HOME?"
    exit 1
  fi
  echo "ANDROID_HOME:$ANDROID_HOME"
}

check_gradle() {
  GRADLE_PATH=$BLACKSMITH_DIR/android/gradlew
  echo "GRADLE_PATH:$GRADLE_PATH"
}

parse_abis_list() {
  # sync with https://github.com/blacksmith/blacksmith/blob/0ca0e02685a9d033ac4f04e2fa5c8ba6dbc5ae50/android/gradle.properties#L1
  ABIS_LIST="armeabi-v7a,arm64-v8a,x86,x86_64"
  CUSTOM_ABIS_LIST=false
  if [ $# -gt 0 ]; then
    ABIS_LIST=$1
    CUSTOM_ABIS_LIST=true
  fi

  echo "ABIS_LIST:$ABIS_LIST"
  echo "CUSTOM_ABIS_LIST:$CUSTOM_ABIS_LIST"
}

build_android() {
  BLACKSMITH_ANDROID_DIR="$BLACKSMITH_DIR/android"
  BUILD_ROOT="${BUILD_ROOT:-$BLACKSMITH_DIR}"
  echo "BUILD_ROOT:$BUILD_ROOT"

  LIB_DIR="$BLACKSMITH_ANDROID_DIR/blacksmith_android/src/main/jniLibs"
  INCLUDE_DIR="$BLACKSMITH_ANDROID_DIR/blacksmith_android/src/main/cpp/libsmith_include"

  # These directories only contain symbolic links.
  rm -rf "$LIB_DIR" && mkdir -p "$LIB_DIR"
  rm -rf "$INCLUDE_DIR" && mkdir -p "$INCLUDE_DIR"

  for abi in $(echo "$ABIS_LIST" | tr ',' '\n')
  do
    echo "abi:$abi"
    ANDROID_BUILD_ROOT="$BUILD_ROOT/build_android_$abi"
    ANDROID_ABI="$abi" \
      BUILD_ROOT="$ANDROID_BUILD_ROOT" \
      "$BLACKSMITH_DIR/scripts/build_android.sh" \
      -DANDROID_CCACHE="$(which ccache)" \
      -DUSE_LITE_INTERPRETER_PROFILER="OFF"

    echo "$abi build output lib,include at $ANDROID_BUILD_ROOT/install"
    ln -s "$ANDROID_BUILD_ROOT/install/lib" "$LIB_DIR/$abi"
    ln -s "$ANDROID_BUILD_ROOT/install/include" "$INCLUDE_DIR/$abi"
  done
}
