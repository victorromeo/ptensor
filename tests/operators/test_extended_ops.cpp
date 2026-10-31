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

// ---------------------------------------------------------------------------
// Fast paths: the same op run on dense tensors (pointer kernels) and with one
// input behind a PackedRomTensor (forces the per-element reference loop) must
// agree. Float32 packing is lossless, so any difference is a kernel bug.
// ---------------------------------------------------------------------------

namespace {

float fa[4096], fb[4096], fbias[256], out_fast[4096], out_ref[4096];

void fill_inputs() {
  uint32_t s = 7;
  for (int i = 0; i < 4096; ++i) {
    s = s * 1664525u + 1013904223u;
    fa[i] = static_cast<float>(static_cast<int32_t>(s >> 9) % 2001) / 1000.0f - 1.0f;
    s = s * 1664525u + 1013904223u;
    fb[i] = static_cast<float>(static_cast<int32_t>(s >> 9) % 2001) / 1000.0f - 1.0f;
  }
  for (int i = 0; i < 256; ++i) fbias[i] = 0.01f * (i % 17) - 0.08f;
}

Tensor dense_or_packed(const TensorShape& s, const float* p, bool packed) {
  if (packed)
    return new PackedRomTensor(s, flt, PackedEncoding::of(PackedFormat::Float32), p);
  return new RomTensor(s, flt, p);
}

void expect_same(uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    ASSERT_NEAR(out_fast[i], out_ref[i], 1e-5f * (1.0f + std::fabs(out_ref[i]))) << "index " << i;
}

}  // namespace

TEST_F(OpsFixture, fast_matmul_matches_reference) {
  fill_inputs();
  for (int pass = 0; pass < 2; ++pass) {
    typedef MatrixMultOperator<float> Op;
    Tensor a = dense_or_packed({7, 13}, fa, pass == 1);
    Tensor b = new RomTensor({13, 5}, flt, fb);
    Tensor c = new BufferTensor({7, 5}, flt, pass ? out_ref : out_fast);
    Op op;
    op.set_inputs({{Op::a, a}, {Op::b, b}}).set_outputs({{Op::c, c}}).eval();
  }
  expect_same(35);
}

TEST_F(OpsFixture, fast_fully_connected_matches_reference) {
  fill_inputs();
  for (int pass = 0; pass < 2; ++pass) {
    typedef FullyConnectedOperator<float> Op;
    Tensor x = dense_or_packed({3, 20}, fa, pass == 1);
    Tensor w = new RomTensor({20, 9}, flt, fb);
    Tensor b = new RomTensor({9}, flt, fbias);
    Tensor y = new BufferTensor({3, 9}, flt, pass ? out_ref : out_fast);
    Op op(Fuseable::ReLU<float>);
    op.set_inputs({{Op::input, x}, {Op::filter, w}, {Op::bias, b}})
        .set_outputs({{Op::output, y}}).eval();
  }
  expect_same(27);
}

TEST_F(OpsFixture, fast_conv2d_matches_reference) {
  fill_inputs();
  const Padding pads[2] = {SAME, VALID};
  for (int pi = 0; pi < 2; ++pi)
    for (uint16_t stride = 1; stride <= 2; ++stride) {
      const uint16_t oh = pads[pi] == SAME ? (9 + stride - 1) / stride : (9 - 3) / stride + 1;
      const uint16_t ow = pads[pi] == SAME ? (8 + stride - 1) / stride : (8 - 3) / stride + 1;
      for (int pass = 0; pass < 2; ++pass) {
        typedef Conv2dOperator<float> Op;
        Tensor x = dense_or_packed({1, 9, 8, 3}, fa, pass == 1);
        Tensor w = new RomTensor({4, 3, 3, 3}, flt, fb);
        Tensor b = new RomTensor({4}, flt, fbias);
        Tensor y = new BufferTensor({1, oh, ow, 4}, flt, pass ? out_ref : out_fast);
        Op op({stride, stride}, pads[pi]);
        op.set_inputs({{Op::in, x}, {Op::filter, w}, {Op::bias, b}})
            .set_outputs({{Op::out, y}}).eval();
      }
      expect_same(oh * ow * 4);
    }
}

TEST_F(OpsFixture, fast_depthwise_matches_reference) {
  fill_inputs();
  for (uint16_t mult = 1; mult <= 2; ++mult) {
    for (int pass = 0; pass < 2; ++pass) {
      typedef DepthwiseConv2dOperator<float> Op;
      Tensor x = dense_or_packed({2, 6, 7, 3}, fa, pass == 1);
      Tensor w = new RomTensor({1, 3, 3, (uint16_t)(3 * mult)}, flt, fb);
      Tensor b = new RomTensor({(uint16_t)(3 * mult)}, flt, fbias);
      Tensor y = new BufferTensor({2, 3, 4, (uint16_t)(3 * mult)}, flt, pass ? out_ref : out_fast);
      Op op(2, 2, SAME, mult, 1, 1);
      op.set_inputs({{Op::in, x}, {Op::filter, w}, {Op::bias, b}})
          .set_outputs({{Op::out, y}}).eval();
    }
    expect_same(2 * 3 * 4 * 3 * mult);
  }
}

TEST_F(OpsFixture, fast_pooling_matches_reference) {
  fill_inputs();
  for (int pass = 0; pass < 2; ++pass) {
    typedef MaxPoolOperator<float> Op;
    Tensor x = dense_or_packed({1, 8, 8, 5}, fa, pass == 1);
    Tensor y = new BufferTensor({1, 4, 4, 5}, flt, pass ? out_ref : out_fast);
    Op op({2, 2}, {1, 2, 2, 1}, SAME);
    op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
  }
  expect_same(80);
  for (int pass = 0; pass < 2; ++pass) {
    typedef AvgPoolOperator<float> Op;
    Tensor x = dense_or_packed({1, 8, 8, 5}, fa, pass == 1);
    Tensor y = new BufferTensor({1, 3, 3, 5}, flt, pass ? out_ref : out_fast);
    Op op({3, 3}, {1, 2, 2, 1}, VALID);
    op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
  }
  expect_same(45);
}

TEST_F(OpsFixture, fast_broadcast_and_unary_match_reference) {
  fill_inputs();
  for (int pass = 0; pass < 2; ++pass) {
    typedef MulOperator<float> Op;
    Tensor a = dense_or_packed({2, 1, 5, 6}, fa, pass == 1);
    Tensor b = new RomTensor({3, 1, 6}, flt, fb);
    Tensor c = new BufferTensor({2, 3, 5, 6}, flt, pass ? out_ref : out_fast);
    Op op;
    op.set_inputs({{Op::a, a}, {Op::b, b}}).set_outputs({{Op::c, c}}).eval();
  }
  expect_same(180);
  for (int pass = 0; pass < 2; ++pass) {
    Tensor a = dense_or_packed({100}, fa, pass == 1);
    Tensor c = new BufferTensor({100}, flt, pass ? out_ref : out_fast);
    GeluOperator op;
    op.set_inputs({{GeluOperator::in, a}}).set_outputs({{GeluOperator::out, c}}).eval();
  }
  expect_same(100);
}

TEST_F(OpsFixture, fast_recurrent_cells_match_reference) {
  fill_inputs();
  // hidden = 20 is not a multiple of the 16-column block.
  const uint16_t B = 2, I = 5, H = 20;
  for (int pass = 0; pass < 2; ++pass) {
    typedef LstmCellOperator Op;
    Tensor x = dense_or_packed({B, I}, fa, pass == 1);
    Tensor h = new RomTensor({B, H}, flt, fa + 100);
    Tensor c = new RomTensor({B, H}, flt, fa + 200);
    Tensor W = new RomTensor({I, 4 * H}, flt, fb);
    Tensor U = new RomTensor({H, 4 * H}, flt, fb + 1000);
    Tensor b = new RomTensor({4 * H}, flt, fbias);
    float* o = pass ? out_ref : out_fast;
    Tensor ho = new BufferTensor({B, H}, flt, o);
    Tensor co = new BufferTensor({B, H}, flt, o + B * H);
    Op op;
    op.set_inputs({{Op::x, x}, {Op::h, h}, {Op::c, c}, {Op::W, W}, {Op::U, U}, {Op::b, b}})
        .set_outputs({{Op::h_out, ho}, {Op::c_out, co}}).eval();
  }
  expect_same(2 * B * H);
  for (int pass = 0; pass < 2; ++pass) {
    typedef GruCellOperator Op;
    Tensor x = dense_or_packed({B, I}, fa, pass == 1);
    Tensor h = new RomTensor({B, H}, flt, fa + 100);
    Tensor W = new RomTensor({I, 3 * H}, flt, fb);
    Tensor U = new RomTensor({H, 3 * H}, flt, fb + 1000);
    Tensor bx = new RomTensor({3 * H}, flt, fbias);
    Tensor bh = new RomTensor({3 * H}, flt, fbias + 60);
    Tensor ho = new BufferTensor({B, H}, flt, pass ? out_ref : out_fast);
    Op op;
    op.set_inputs({{Op::x, x}, {Op::h, h}, {Op::W, W}, {Op::U, U}, {Op::bx, bx}, {Op::bh, bh}})
        .set_outputs({{Op::h_out, ho}}).eval();
  }
  expect_same(B * H);
  EXPECT_EQ(errors, 0);
}
