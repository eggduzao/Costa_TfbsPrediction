// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <ATen/core/type_factory.h>
#include "caffe2/android/blacksmith_android/src/main/cpp/blacksmith_jni_common.h"

using namespace ::testing;

TEST(blacksmith_jni_common_test, newJIValueFromAtIValue) {
  auto dict = c10::impl::GenericDict(
      c10::dynT<c10::IntType>(), c10::dynT<c10::StringType>());
  auto dictCallback = [](auto&&) {
    return facebook::jni::local_ref<blacksmith_jni::JIValue>{};
  };
  EXPECT_NO_THROW(blacksmith_jni::JIValue::newJIValueFromAtIValue(
      dict, dictCallback, dictCallback));
}
