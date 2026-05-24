#include <gtest/gtest.h>
#include "tt/tt.hpp"

using tt::dtype;

TEST(TensorTest, ShapeAndDimAndSize) {
  auto t = tt::from<dtype::f32, 2, 3>();
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.size(), 6);
}

TEST(TensorTest, ZerosAndOnes) {
  auto zeros = tt::zeros<dtype::u8, 1, 100>();
  for (const auto n : zeros) {
    EXPECT_EQ(n, 0);
  }

  auto ones = tt::ones<dtype::f64, 2, 42>();
  for (const auto n : ones) {
    EXPECT_EQ(n, 1.0f);
  }
}
