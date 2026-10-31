// C++-side checks for the extended operators. The float paths are checked
// exhaustively against NumPy in python/tests/test_layers.py; these cover the
// integer instantiations, error paths and use with packed weights.
#include <cmath>

#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

struct OpsFixture : public ::testing::Test {
  localCircularArenaAllocator<4096> meta_allocator;
  localCircularArenaAllocator<4096, uint32_t> ram_allocator;
  SimpleErrorHandler err{10};
  int errors = 0;

  void SetUp() override {
    err.set_onError([this](Error*) { ++errors; });
    Context::get_default_context()->set_ErrorHandler(&err);
    Context::get_default_context()->set_metadata_allocator(&meta_allocator);
    Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  }
  void TearDown() override {
    err.set_onError([](Error*) {});
  }
};

}  // namespace

TEST_F(OpsFixture, int16_broadcast_mul_row_by_column) {
  const int16_t a[3] = {1, 2, 3};    // [3, 1]
  const int16_t b[2] = {10, -20};    // [2]
  int16_t out[6] = {0};
  {
    Tensor ta = new RomTensor({3, 1}, i16, a);
    Tensor tb = new RomTensor({2}, i16, b);
    Tensor tc = new BufferTensor({3, 2}, i16, out);
    MulOperator<int16_t> op;
    op.set_inputs({{MulOperator<int16_t>::a, ta}, {MulOperator<int16_t>::b, tb}})
        .set_outputs({{MulOperator<int16_t>::c, tc}})
        .eval();
  }
  const int16_t expect[6] = {10, -20, 20, -40, 30, -60};
  for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], expect[i]) << i;
  EXPECT_EQ(errors, 0);
}

TEST_F(OpsFixture, uint8_maximum_equal_shapes) {
  const uint8_t a[4] = {1, 200, 3, 4};
  const uint8_t b[4] = {9, 100, 3, 0};
  uint8_t out[4] = {0};
  {
    typedef MaximumOperator<uint8_t> Op;
    Tensor ta = new RomTensor({4}, u8, a);
    Tensor tb = new RomTensor({4}, u8, b);
    Tensor tc = new BufferTensor({4}, u8, out);
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  EXPECT_EQ(out[0], 9);
  EXPECT_EQ(out[1], 200);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 4);
}

TEST_F(OpsFixture, broadcast_mismatch_raises_error) {
  const float a[3] = {1, 2, 3};
  const float b[2] = {1, 2};
  float out[3] = {0};
  {
    typedef SubOperator<float> Op;
    Tensor ta = new RomTensor({3}, flt, a);
    Tensor tb = new RomTensor({2}, flt, b);
    Tensor tc = new BufferTensor({3}, flt, out);
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  EXPECT_GT(errors, 0);
}

TEST_F(OpsFixture, leaky_relu_carries_its_alpha) {
  const float in[4] = {-2, -1, 0, 3};
  float out[4] = {0};
  {
    Tensor ti = new RomTensor({4}, flt, in);
    Tensor to = new BufferTensor({4}, flt, out);
    LeakyReLUOperator op(activations::LeakyReLU(0.5f));
    op.set_inputs({{LeakyReLUOperator::in, ti}})
        .set_outputs({{LeakyReLUOperator::out, to}})
        .eval();
  }
  EXPECT_FLOAT_EQ(out[0], -1.0f);
  EXPECT_FLOAT_EQ(out[1], -0.5f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 3.0f);
}

TEST_F(OpsFixture, int8_concat_along_last_axis) {
  const int8_t a[4] = {1, 2, 3, 4};  // [2, 2]
  const int8_t b[2] = {-1, -2};      // [2, 1]
  int8_t out[6] = {0};
  {
    typedef ConcatOperator<int8_t> Op;
    Tensor ta = new RomTensor({2, 2}, i8, a);
    Tensor tb = new RomTensor({2, 1}, i8, b);
    Tensor to = new BufferTensor({2, 3}, i8, out);
    Op op(1);
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::out, to}}).eval();
  }
  const int8_t expect[6] = {1, 2, -1, 3, 4, -2};
  for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], expect[i]) << i;
}

TEST_F(OpsFixture, depthwise_conv_reads_6bit_packed_weights) {
  // 1x3x3x1 input, 3x3 all-ones filter stored as 6-bit codes (scale 0.5,
  // zero point 30: code 32 -> 1.0), VALID padding -> one output = sum(input).
  const float in[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint32_t codes[9];
  for (int i = 0; i < 9; ++i) codes[i] = 32;
  uint8_t packed[8] = {0};
  packing::pack_bits(codes, 9, 6, packed);
  float out[1] = {0};
  {
    typedef DepthwiseConv2dOperator<float> Op;
    Tensor ti = new RomTensor({1, 3, 3, 1}, flt, in);
    Tensor tf = new PackedRomTensor({1, 3, 3, 1}, flt,
                                    PackedEncoding::bits(6, 0.5f, 30), packed);
    Tensor to = new BufferTensor({1, 1, 1, 1}, flt, out);
    Op op(1, 1, VALID, 1, 1, 1);
    op.set_inputs({{Op::in, ti}, {Op::filter, tf}}).set_outputs({{Op::out, to}}).eval();
  }
  EXPECT_FLOAT_EQ(out[0], 45.0f);
  EXPECT_EQ(errors, 0);
}
