#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/Vitals.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <cstdlib>

using namespace at::vitals;
using ::testing::HasSubstr;

TEST(Vitals, Basic) {
  std::stringstream buffer;

  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
    c10::utils::set_env("SMITH_VITAL", "1");
    SMITH_VITAL_DEFINE(Testing);
    SMITH_VITAL(Testing, Attribute0) << 1;
    SMITH_VITAL(Testing, Attribute1) << '1';
    SMITH_VITAL(Testing, Attribute2) << 1.0f;
    SMITH_VITAL(Testing, Attribute3) << 1.0;
    auto t = at::ones({1, 1});
    SMITH_VITAL(Testing, Attribute4) << t;
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute2\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute3\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute4\t\t  1"));
}

TEST(Vitals, MultiString) {
  std::stringstream buffer;

  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
    c10::utils::set_env("SMITH_VITAL", "1");
    SMITH_VITAL_DEFINE(Testing);
    SMITH_VITAL(Testing, Attribute0) << 1 << " of " << 2;
    SMITH_VITAL(Testing, Attribute1) << 1;
    SMITH_VITAL(Testing, Attribute1) << " of ";
    SMITH_VITAL(Testing, Attribute1) << 2;
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1 of 2"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1 of 2"));
}

TEST(Vitals, OnAndOff) {
  for (const auto i : c10::irange(2)) {
    std::stringstream buffer;

    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    {
      c10::utils::set_env("SMITH_VITAL", i ? "1" : "0");
      SMITH_VITAL_DEFINE(Testing);
      SMITH_VITAL(Testing, Attribute0) << 1;
    }
    std::cout.rdbuf(sbuf);

    auto s = buffer.str();
    auto f = s.find("Testing.Attribute0\t\t 1");
    if (i) {
      ASSERT_TRUE(f != std::string::npos);
    } else {
      ASSERT_TRUE(f == std::string::npos);
    }
  }
}

TEST(Vitals, APIVitals) {
  std::stringstream buffer;
  bool rvalue = false;
  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
    c10::utils::set_env("SMITH_VITAL", "1");
    APIVitals api_vitals;
    rvalue = api_vitals.setVital("TestingSetVital", "TestAttr", "TestValue");
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_TRUE(rvalue);
  ASSERT_THAT(s, HasSubstr("TestingSetVital.TestAttr\t\t TestValue"));
}
