#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

struct PolyFixture : public ::testing::Test {
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

// Reference value written from the format description (double precision).
double reference_value(const float* c, int degree, int bits, uint32_t k) {
  const double t = 2.0 * k / ((1u << bits) - 1u) - 1.0;
  double v = 0, p = 1;
  for (int i = 0; i <= degree; ++i, p *= t) v += c[i] * p;
  return v;
}

std::vector<uint8_t> pack(const std::vector<uint32_t>& codes, uint8_t bits) {
  std::vector<uint8_t> out(packing::packed_size_bytes(codes.size(), bits) + 1);
  packing::pack_bits(codes.data(), codes.size(), bits, out.data());
  return out;
}

}  // namespace

TEST(PolyQuant, code_to_t_spans_minus_one_to_one) {
  for (uint8_t bits = 1; bits <= 16; ++bits) {
    EXPECT_FLOAT_EQ(polyquant::code_to_t(0, bits), -1.0f);
    EXPECT_FLOAT_EQ(polyquant::code_to_t((1u << bits) - 1u, bits), 1.0f);
  }
  EXPECT_FLOAT_EQ(polyquant::code_to_t(1, 2), -1.0f / 3.0f);
}

TEST(PolyQuant, lut_matches_the_polynomial) {
  const float c[4] = {0.01f, 0.05f, -0.02f, 0.08f};
  for (int degree = 1; degree <= 3; ++degree) {
    for (uint8_t bits : {2, 4, 8}) {
      std::vector<float> lut(1u << bits);
      polyquant::build_lut(c, degree, bits, lut.data());
      for (uint32_t k = 0; k < lut.size(); ++k)
        EXPECT_NEAR(lut[k], reference_value(c, degree, bits, k), 1e-6) << k;
    }
  }
}

TEST_F(PolyFixture, decodes_with_and_without_a_lut) {
  const float c[4] = {-0.003f, 0.11f, 0.004f, 0.09f};
  const uint16_t shapes[][2] = {{6, 8}, {5, 7}, {1, 1}};
  for (uint8_t bits : {2, 3, 5, 8, 12, 16}) {
    for (int degree = 1; degree <= 3; ++degree) {
      std::vector<float> lut;
      const bool with_lut = bits <= 8;
      if (with_lut) {
        lut.resize(1u << bits);
        polyquant::build_lut(c, degree, bits, lut.data());
      }
      PolyEncoding e = PolyEncoding::make(degree, bits, c,
                                          with_lut ? lut.data() : nullptr);
      for (auto& sh : shapes) {
        const uint32_t count = sh[0] * sh[1];
        std::vector<uint32_t> codes(count);
        for (uint32_t i = 0; i < count; ++i)
          codes[i] = (i * 2654435761u) & ((1u << bits) - 1u);
        std::vector<uint8_t> stream = pack(codes, bits);
        EXPECT_EQ(poly_stream_size(e, count), stream.size() - 1);
        Tensor t = new PolyRomTensor({sh[0], sh[1]}, e, stream.data());
        const Tensor& ct = t;
        for (uint32_t j = 0; j < count; ++j) {
          const uint32_t i = (j * 7u) % count;
          EXPECT_NEAR(static_cast<float>(ct(i)),
                      reference_value(c, degree, bits, codes[i]), 1e-6)
              << int(bits) << "/" << degree << "/" << i;
        }
        std::vector<float> all(count);
        static_cast<PolyRomTensor*>(*t)->decode_all(all.data());
        for (uint32_t i = 0; i < count; ++i)
          EXPECT_EQ(all[i], polyquant::decode_code(c, degree, bits, codes[i]));
      }
    }
  }
  EXPECT_FALSE(got_error);
}

TEST_F(PolyFixture, fully_connected_matches_decoded_float_weights) {
  const float c[4] = {0.0f, 0.2f, 0.0f, 0.3f};
  const uint8_t bits = 4;
  const uint16_t in_f = 12, out_f = 5;
  const uint32_t count = in_f * out_f;
  std::vector<float> lut(1u << bits);
  polyquant::build_lut(c, 3, bits, lut.data());
  std::vector<uint32_t> codes(count);
  for (uint32_t i = 0; i < count; ++i) codes[i] = (i * 40503u + 11u) & 15u;
  std::vector<uint8_t> stream = pack(codes, bits);
  PolyEncoding e = PolyEncoding::make(3, bits, c, lut.data());

  std::vector<float> wf(count), x(in_f), bias(out_f), y_p(out_f), y_f(out_f);
  for (uint32_t i = 0; i < count; ++i) wf[i] = lut[codes[i]];
  for (int i = 0; i < in_f; ++i) x[i] = 0.1f * i - 0.4f;
  for (int o = 0; o < out_f; ++o) bias[o] = 0.01f * o;

  typedef FullyConnectedOperator<float> FC;
  {
    Tensor tx = new RomTensor({1, in_f}, flt, x.data());
    Tensor tw = new PolyRomTensor({in_f, out_f}, e, stream.data());
    Tensor tb = new RomTensor({out_f}, flt, bias.data());
    Tensor ty = new BufferTensor({1, out_f}, flt, y_p.data());
    FC op;
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}})
        .set_outputs({{FC::output, ty}}).eval();
  }
  {
    Tensor tx = new RomTensor({1, in_f}, flt, x.data());
    Tensor tw = new RomTensor({in_f, out_f}, flt, wf.data());
    Tensor tb = new RomTensor({out_f}, flt, bias.data());
    Tensor ty = new BufferTensor({1, out_f}, flt, y_f.data());
    FC op;
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}})
        .set_outputs({{FC::output, ty}}).eval();
  }
  for (int o = 0; o < out_f; ++o) EXPECT_NEAR(y_p[o], y_f[o], 1e-5f);
  EXPECT_FALSE(got_error);
}

TEST_F(PolyFixture, rejects_bad_encodings) {
  const float c[4] = {0, 1, 0, 0};
  uint8_t stream[4] = {0};
  Tensor a = new PolyRomTensor({2}, PolyEncoding::make(0, 4, c), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor b = new PolyRomTensor({2}, PolyEncoding::make(4, 4, c), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor d = new PolyRomTensor({2}, PolyEncoding::make(2, 17, c), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor e = new PolyRomTensor({2}, PolyEncoding::make(2, 4, c), stream);
  EXPECT_FALSE(got_error);
}
