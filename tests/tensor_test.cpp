#include <gtest/gtest.h>
#include "tt/tt.hpp"

using namespace tt;

TEST(TensorTest, ShapeAndDimAndSize) {
  Tensor<dtype::f32, 2, 3> t;
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.dim, 2);
  EXPECT_EQ(t.size(), 6);
}

TEST(TensorTest, ZerosAndOnes) {
  auto zeros = Tensor<dtype::f32, 1, 100>::zeros();
  for (const auto n : zeros) {
    EXPECT_EQ(n, 0.0);
  }

  auto ones = Tensor<dtype::f64, 1, 100>::ones();
  for (const auto n : ones) {
    EXPECT_EQ(n, 1.0);
  }
}
