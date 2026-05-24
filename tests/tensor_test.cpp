#include <gtest/gtest.h>
#include "tt/tt.hpp"

using enum tt::dtype;

TEST(TensorTest, StaticZeros) {
  auto t = tt::zeros<f32, 2, 3>();
  EXPECT_EQ(t.size(), 6);
  for (size_t i = 0; i < t.size(); ++i)
      EXPECT_FLOAT_EQ(t[i], 0.0f);
}

TEST(TensorTest, DynamicOnes) {
  auto t = tt::ones<f64>({2, 3});
  EXPECT_EQ(t.size(), 6);
  for (size_t i = 0; i < t.size(); ++i)
      EXPECT_DOUBLE_EQ(t[i], 1.0);
}

TEST(TensorTest, StaticFrom) {
  auto t = tt::from<f32, 2, 3>({1, 2, 3, 4, 5, 6});
  EXPECT_FLOAT_EQ(t[0], 1.0f);
  EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(TensorTest, StaticArange) {
  auto t = tt::arange<f32, 5>(tt::Range<int>{.end = 5});
  for (size_t i = 0; i < t.size(); ++i)
      EXPECT_FLOAT_EQ(t[i], static_cast<float>(i));
}

TEST(TensorTest, DynamicArange) {
  auto t = tt::arange<i64>(tt::Range<int>{.end = 50});
  EXPECT_EQ(t.size(), 50);
  for (size_t i = 0; i < t.size(); ++i)
      EXPECT_EQ(t[i], static_cast<int64_t>(i));
}

TEST(TensorTest, ArangeWithStep) {
  auto t = tt::arange<f32>(tt::Range<float>{.start = 0, .end = 1, .step = 0.5f});
  EXPECT_EQ(t.size(), 2);
  EXPECT_FLOAT_EQ(t[0], 0.0f);
  EXPECT_FLOAT_EQ(t[1], 0.5f);
}

TEST(TensorTest, StaticShape) {
  auto t = tt::zeros<f32, 2, 3>();
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
}

TEST(TensorTest, DynamicShape) {
  auto t = tt::zeros<f64>({4, 5});
  EXPECT_EQ(t.shape[0], 4);
  EXPECT_EQ(t.shape[1], 5);
}

TEST(TensorTest, StaticFromSizeMismatch) {
  EXPECT_THROW((tt::from<f32, 2, 3>({1, 2, 3})), std::invalid_argument);
}
