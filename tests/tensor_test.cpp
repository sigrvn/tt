#include <gtest/gtest.h>
#include "tt/tt.hpp"

using namespace tt;

TEST(TensorTest, ShapeAndDimAndSize) {
  Tensor<2, 3> t;
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.dim, 2);
  EXPECT_EQ(t.size(), 6);
}
