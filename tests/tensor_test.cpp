#include <gtest/gtest.h>
#include "tt/tt.hpp"

using namespace tt::prelude;

TEST(Static, Zeros) {
  auto t = tt::zeros<f32>(Shape<2,3>{});
  EXPECT_EQ(t.numel(), 6);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_FLOAT_EQ(t[i], 0.0f);
}

TEST(Dynamic, Zeros) {
  auto t = tt::zeros<f32>({2,3});
  EXPECT_EQ(t.numel(), 6);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_FLOAT_EQ(t[i], 0.0f);
}

TEST(Static, Ones) {
  auto t = tt::ones<f64>(Shape<2,3>{});
  EXPECT_EQ(t.numel(), 6);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_DOUBLE_EQ(t[i], 1.0);
}

TEST(Dynamic, Ones) {
  auto t = tt::ones<f64>({2,3});
  EXPECT_EQ(t.numel(), 6);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_DOUBLE_EQ(t[i], 1.0);
}

TEST(Static, From) {
  auto t = tt::from<f32>(Shape<2,3>{}, {1, 2, 3, 4, 5, 6});
  EXPECT_FLOAT_EQ(t[0], 1.0f);
  EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(Static, FromSizeMismatch) {
  EXPECT_THROW((tt::from<f32>(Shape<2,3>{}, {1, 2, 3})), std::invalid_argument);
}

TEST(Dynamic, From) {
  auto t = tt::from<f32>({2,3}, {1, 2, 3, 4, 5, 6});
  EXPECT_FLOAT_EQ(t[0], 1.0f);
  EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(Static, Full) {
  auto t = tt::full<f32>(Shape<2,3>{}, 7.0f);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_FLOAT_EQ(t[i], 7.0f);
}

TEST(Static, Arange) {
  auto t = tt::arange<f32>(Shape<5>{}, Range{.end = 5});
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_FLOAT_EQ(t[i], static_cast<float>(i));
}

TEST(Dynamic, Arange) {
  auto t = tt::arange<i64>(Range{.end = 50});
  EXPECT_EQ(t.numel(), 50);
  for (size_t i = 0; i < t.numel(); ++i)
    EXPECT_EQ(t[i], static_cast<std::int64_t>(i));
}

TEST(Dynamic, ArangeWithFloatStep) {
  auto t = tt::arange<f32>(Range<float>{.start = 0, .end = 1, .step = 0.5f});
  EXPECT_EQ(t.numel(), 2);
  EXPECT_FLOAT_EQ(t[0], 0.0f);
  EXPECT_FLOAT_EQ(t[1], 0.5f);
}

TEST(Static, Shape) {
  auto t = tt::zeros<f32>(Shape<2,3>{});
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
}

TEST(Dynamic, Shape) {
  auto t = tt::zeros<f64>({4,5});
  EXPECT_EQ(t.shape[0], 4);
  EXPECT_EQ(t.shape[1], 5);
}

TEST(TensorTest, StaticAdd) {
  const auto a = tt::full<f32>(Shape<2, 3>{}, 1.0f);
  const auto b = tt::full<f32>(Shape<2, 3>{}, 2.0f);
  auto c = tt::add(a, b);
  for (size_t i = 0; i < c.numel(); ++i)
    EXPECT_FLOAT_EQ(c[i], 3.0f);
}

TEST(TensorTest, DynamicAddShapeMismatch) {
  auto a = tt::zeros<f32>({2, 2});
  auto b = tt::zeros<f32>({2, 3});
  EXPECT_THROW(tt::add(a, b), std::invalid_argument);
}
