#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

struct ArenaFixture : public ::testing::Test {
  localCircularArenaAllocator<2048> meta_allocator;
  localCircularArenaAllocator<4096, uint32_t> ram_allocator;
  SimpleErrorHandler err{10};
  bool got_error = false;

  void SetUp() override {
    err.set_onError([this](Error*) { got_error = true; });
    Context::get_default_context()->set_ErrorHandler(&err);
    Context::get_default_context()->set_metadata_allocator(&meta_allocator);
    Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  }
  void TearDown() override {
    Context::get_default_context()->set_ErrorHandler(nullptr);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Bit stream packing
// ---------------------------------------------------------------------------

TEST(Packing, every_width_round_trips) {
  for (uint8_t bits = 1; bits <= 16; ++bits) {
    const size_t n = 97;  // odd count so the stream never ends on a byte edge
    std::vector<uint32_t> codes(n);
    const uint32_t max_code = (1u << bits) - 1u;
    for (size_t i = 0; i < n; ++i) codes[i] = (i * 2654435761u) & max_code;
    codes[0] = 0;
    codes[n - 1] = max_code;
    std::vector<uint8_t> buf(packing::packed_size_bytes(n, bits) + 3, 0xEE);
    packing::pack_bits(codes.data(), n, bits, buf.data());
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(packing::read_bits(buf.data(), i, bits), codes[i])
          << "bits=" << int(bits) << " index=" << i;
    }
    // Packing must not write past the computed size.
    for (size_t i = packing::packed_size_bytes(n, bits); i < buf.size(); ++i) {
      EXPECT_EQ(buf[i], 0xEE);
    }
  }
}

TEST(Packing, six_and_seven_bit_are_dense) {
  EXPECT_EQ(packing::packed_size_bytes(8, 6), 6u);
  EXPECT_EQ(packing::packed_size_bytes(8, 7), 7u);
  EXPECT_EQ(packing::packed_size_bytes(1000, 6), 750u);
}

TEST(Packing, half_precision_known_values) {
  EXPECT_EQ(packing::float_to_half(0.0f), 0x0000);
  EXPECT_EQ(packing::float_to_half(-0.0f), 0x8000);
  EXPECT_EQ(packing::float_to_half(1.0f), 0x3C00);
  EXPECT_EQ(packing::float_to_half(-2.0f), 0xC000);
  EXPECT_EQ(packing::float_to_half(65504.0f), 0x7BFF);  // largest finite
  EXPECT_EQ(packing::float_to_half(1e6f), 0x7C00);      // overflow -> inf
  EXPECT_EQ(packing::float_to_half(5.9604645e-8f), 0x0001);  // min subnormal
  EXPECT_EQ(packing::float_to_half(6.1035156e-5f), 0x0400);  // min normal
  EXPECT_FLOAT_EQ(packing::half_to_float(0x3555), 0.33325195f);
  EXPECT_FLOAT_EQ(packing::half_to_float(0x0001), 5.9604645e-8f);
  EXPECT_TRUE(std::isinf(packing::half_to_float(0x7C00)));
  EXPECT_TRUE(std::isnan(packing::half_to_float(0x7E00)));
}

TEST(Packing, half_round_trips_all_finite_values) {
  for (uint32_t h = 0; h < 0x10000u; ++h) {
    const uint16_t hv = static_cast<uint16_t>(h);
    if ((hv & 0x7C00u) == 0x7C00u) continue;  // inf / nan
    const float f = packing::half_to_float(hv);
    ASSERT_EQ(packing::float_to_half(f), hv) << std::hex << h;
  }
}

// ---------------------------------------------------------------------------
// PackedRomTensor decoding
// ---------------------------------------------------------------------------

typedef ArenaFixture PackedRomTensorTest;

TEST_F(PackedRomTensorTest, six_bit_affine_to_float) {
  // 8 six-bit codes 0..63 packed into 6 bytes.
  const uint32_t codes[8] = {0, 1, 31, 32, 33, 62, 63, 17};
  uint8_t packed[6];
  packing::pack_bits(codes, 8, 6, packed);

  const float scale = 0.05f;
  const int32_t zp = 32;
  Tensor t = new PackedRomTensor({2, 4}, flt, PackedEncoding::bits(6, scale, zp),
                                 packed);
  for (int i = 0; i < 8; ++i) {
    const float expect = (static_cast<int>(codes[i]) - zp) * scale;
    EXPECT_FLOAT_EQ(static_cast<float>(t(i)), expect) << i;
  }
  EXPECT_FALSE(got_error);
}

TEST_F(PackedRomTensorTest, seven_bit_to_uint8_codes) {
  uint32_t codes[13];
  for (int i = 0; i < 13; ++i) codes[i] = (i * 37) & 0x7F;
  uint8_t packed[12];  // ceil(13 * 7 / 8) = 12
  ASSERT_EQ(packing::packed_size_bytes(13, 7), sizeof(packed));
  packing::pack_bits(codes, 13, 7, packed);
  Tensor t = new PackedRomTensor({13}, u8, PackedEncoding::bits(7), packed);
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(static_cast<uint8_t>(t(i)), codes[i]);
  }
}

TEST_F(PackedRomTensorTest, six_bit_codes_as_int8_weights) {
  // Store int8 weights in [-128, 124] at a step of 4 as 6-bit codes:
  // weight = (code - 32) * 4.
  const uint32_t codes[4] = {0, 32, 63, 40};
  uint8_t packed[3];
  packing::pack_bits(codes, 4, 6, packed);
  Tensor t = new PackedRomTensor({4}, i8, PackedEncoding::bits(6, 4.0f, 32),
                                 packed);
  EXPECT_EQ(static_cast<int8_t>(t(0)), -128);
  EXPECT_EQ(static_cast<int8_t>(t(1)), 0);
  EXPECT_EQ(static_cast<int8_t>(t(2)), 124);
  EXPECT_EQ(static_cast<int8_t>(t(3)), 32);
}

TEST_F(PackedRomTensorTest, int16_uint16_storage) {
  const int16_t s16v[3] = {-32768, -1, 32767};
  const uint16_t u16v[3] = {0, 1234, 65535};
  uint8_t sbuf[6], ubuf[6];
  memcpy(sbuf, s16v, sizeof(s16v));  // host is little endian in CI
  memcpy(ubuf, u16v, sizeof(u16v));
  Tensor ts = new PackedRomTensor({3}, i16, PackedEncoding::of(PackedFormat::Int16), sbuf);
  Tensor tu = new PackedRomTensor({3}, u16, PackedEncoding::of(PackedFormat::UInt16), ubuf);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(static_cast<int16_t>(ts(i)), s16v[i]);
    EXPECT_EQ(static_cast<uint16_t>(tu(i)), u16v[i]);
  }
  // Same int16 data, dequantized to float.
  Tensor tf = new PackedRomTensor({3}, flt,
                                  PackedEncoding::of(PackedFormat::Int16, 1.0f / 32768.0f, 0),
                                  sbuf);
  EXPECT_FLOAT_EQ(static_cast<float>(tf(0)), -1.0f);
  EXPECT_NEAR(static_cast<float>(tf(2)), 1.0f, 1e-4f);
}

TEST_F(PackedRomTensorTest, float16_float32_float64_storage) {
  const float values[5] = {0.0f, 1.5f, -3.25f, 1000.0f, 0.1f};
  uint16_t halfs[5];
  double doubles[5];
  for (int i = 0; i < 5; ++i) {
    halfs[i] = packing::float_to_half(values[i]);
    doubles[i] = values[i];
  }
  Tensor th = new PackedRomTensor({5}, flt, PackedEncoding::of(PackedFormat::Float16), halfs);
  Tensor tf = new PackedRomTensor({5}, flt, PackedEncoding::of(PackedFormat::Float32), values);
  Tensor td = new PackedRomTensor({5}, flt, PackedEncoding::of(PackedFormat::Float64), doubles);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(static_cast<float>(th(i)), values[i], std::fabs(values[i]) * 1e-3f);
    EXPECT_FLOAT_EQ(static_cast<float>(tf(i)), values[i]);
    EXPECT_FLOAT_EQ(static_cast<float>(td(i)), values[i]);
  }
}

TEST_F(PackedRomTensorTest, integer_outputs_saturate) {
  const uint32_t codes[2] = {0, 255};
  uint8_t packed[2];
  packing::pack_bits(codes, 2, 8, packed);
  // (code - 128) * 2 spans [-256, 254], beyond int8.
  Tensor t = new PackedRomTensor({2}, i8, PackedEncoding::bits(8, 2.0f, 128), packed);
  EXPECT_EQ(static_cast<int8_t>(t(0)), -128);
  EXPECT_EQ(static_cast<int8_t>(t(1)), 127);
}

TEST_F(PackedRomTensorTest, several_reads_held_at_once) {
  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor t = new PackedRomTensor({4}, flt, PackedEncoding::of(PackedFormat::Float32), values);
  const Tensor& ct = t;
  // Hold all four proxies before converting any of them.
  IntegralValue a = ct(0), b = ct(1), c = ct(2), d = ct(3);
  EXPECT_FLOAT_EQ(static_cast<float>(a) + static_cast<float>(b) +
                      static_cast<float>(c) + static_cast<float>(d),
                  10.0f);
}

TEST_F(PackedRomTensorTest, writes_are_discarded_and_resize_rejected) {
  const float values[2] = {1.0f, 2.0f};
  Tensor t = new PackedRomTensor({2}, flt, PackedEncoding::of(PackedFormat::Float32), values);
  t(0) = 5.0f;
  EXPECT_FLOAT_EQ(static_cast<float>(t(0)), 1.0f);
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FALSE(got_error);
  t->resize({1});
  EXPECT_TRUE(got_error);
}

TEST_F(PackedRomTensorTest, invalid_bit_width_is_rejected) {
  uint8_t dummy[4] = {0};
  Tensor t = new PackedRomTensor({2}, u8, PackedEncoding::bits(17), dummy);
  EXPECT_TRUE(got_error);
}

TEST_F(PackedRomTensorTest, buffer_size_helper) {
  EXPECT_EQ(packed_buffer_size(PackedEncoding::bits(6), 100), 75u);
  EXPECT_EQ(packed_buffer_size(PackedEncoding::of(PackedFormat::Float16), 10), 20u);
  EXPECT_EQ(packed_buffer_size(PackedEncoding::of(PackedFormat::Float64), 3), 24u);
}

// End to end: an operator consumes 6-bit packed weights exactly as it would a
// regular float tensor.
TEST_F(PackedRomTensorTest, matmul_with_six_bit_weights) {
  const float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  // Weights {0.5, -0.25, 1.0, 0.0} as 6-bit codes, scale 0.25, zero point 32.
  const uint32_t codes[4] = {34, 31, 36, 32};
  uint8_t packed[3];
  packing::pack_bits(codes, 4, 6, packed);

  Tensor a = new RomTensor({2, 2}, flt, a_vals);
  Tensor w = new PackedRomTensor({2, 2}, flt, PackedEncoding::bits(6, 0.25f, 32), packed);
  Tensor out = new RamTensor({2, 2}, flt);

  MatrixMultOperator<float> mm;
  mm.set_inputs({{MatrixMultOperator<float>::a, a}, {MatrixMultOperator<float>::b, w}})
      .set_outputs({{MatrixMultOperator<float>::c, out}})
      .eval();

  // [1 2; 3 4] x [0.5 -0.25; 1 0]
  const float expect[4] = {2.5f, -0.25f, 5.5f, -0.75f};
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(static_cast<float>(out(i)), expect[i]);
  EXPECT_FALSE(got_error);
}
