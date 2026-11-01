#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

struct HoughFixture : public ::testing::Test {
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

// A small codebook: 4 lines given by Hough (theta, rho) and a segment.
void make_lines(std::vector<float>& lines, uint8_t t_bits) {
  const double thetas[4] = {0.3, 1.2, 2.0, 2.9};
  const double rhos[4] = {0.1, -0.2, 0.05, 0.3};
  const double levels = (1u << t_bits) - 1u;
  lines.clear();
  for (int k = 0; k < 4; ++k) {
    const double c = std::cos(thetas[k]), s = std::sin(thetas[k]);
    const double t0 = -0.5, step = 1.0 / levels;
    lines.push_back(static_cast<float>(rhos[k] * c - t0 * s));
    lines.push_back(static_cast<float>(rhos[k] * s + t0 * c));
    lines.push_back(static_cast<float>(-step * s));
    lines.push_back(static_cast<float>(step * c));
  }
}

// Reference decoder written from the format description, independent of hough.hpp.
std::vector<float> reference_decode(const std::vector<uint32_t>& codes,
                                    const std::vector<float>& lines,
                                    uint8_t line_bits, uint32_t stride,
                                    uint32_t count) {
  std::vector<float> out(count, 0.0f);
  std::vector<std::pair<uint32_t, uint32_t> > slots;  // (x index, y index)
  const uint32_t block = 2 * stride;
  const uint32_t full = count / block;
  for (uint32_t b = 0; b < full; ++b)
    for (uint32_t r = 0; r < stride; ++r)
      slots.push_back(std::make_pair(b * block + r, b * block + r + stride));
  for (uint32_t i = full * block; i < count; i += 2)
    slots.push_back(std::make_pair(i, i + 1));
  for (size_t p = 0; p < slots.size(); ++p) {
    const uint32_t k = codes[p] & ((1u << line_bits) - 1u);
    const float tc = static_cast<float>(codes[p] >> line_bits);
    const float* l = &lines[k * 4];
    out[slots[p].first] = l[0] + tc * l[2];
    if (slots[p].second < count) out[slots[p].second] = l[1] + tc * l[3];
  }
  return out;
}

}  // namespace

TEST(Hough, pair_mapping_covers_every_value_once) {
  const uint32_t counts[] = {1, 2, 3, 7, 8, 9, 31, 64, 100};
  const uint32_t strides[] = {1, 2, 3, 4, 16};
  for (uint32_t count : counts) {
    for (uint32_t stride : strides) {
      const uint32_t pairs = hough::pair_count(count, stride);
      std::vector<int> seen(2 * pairs, 0);
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t comp;
        const uint32_t p = hough::pair_of(i, count, stride, &comp);
        ASSERT_LT(p, pairs) << count << "/" << stride;
        seen[2 * p + comp]++;
      }
      int total = 0;
      for (int v : seen) {
        ASSERT_LE(v, 1);
        total += v;
      }
      EXPECT_EQ(total, static_cast<int>(count));
      EXPECT_LE(2 * pairs - count, 1u) << "at most one unused slot";
    }
  }
}

TEST_F(HoughFixture, decodes_like_the_reference) {
  const uint8_t line_bits = 2, t_bits = 5;
  std::vector<float> lines;
  make_lines(lines, t_bits);
  const uint32_t shapes[][2] = {{6, 8}, {5, 7}, {1, 1}};
  const uint16_t strides[] = {1, 8, 3};
  for (int si = 0; si < 3; ++si) {
    const uint32_t count = shapes[si][0] * shapes[si][1];
    const uint16_t stride = strides[si];
    const uint32_t pairs = hough::pair_count(count, stride);
    std::vector<uint32_t> codes(pairs);
    for (uint32_t p = 0; p < pairs; ++p) codes[p] = (p * 2654435761u) & 127u;
    std::vector<uint8_t> stream(hough::stream_bytes(count, stride, line_bits, t_bits));
    packing::pack_bits(codes.data(), pairs, line_bits + t_bits, stream.data());
    const std::vector<float> expect =
        reference_decode(codes, lines, line_bits, stride, count);

    HoughEncoding e = HoughEncoding::make(line_bits, t_bits, stride, lines.data());
    EXPECT_EQ(hough_stream_size(e, count), stream.size());
    Tensor t = new HoughRomTensor({(uint16_t)shapes[si][0], (uint16_t)shapes[si][1]}, e,
                                  stream.data());
    const Tensor& ct = t;
    // Read in a scattered order so the pair cache is exercised both ways.
    for (uint32_t j = 0; j < count; ++j) {
      const uint32_t i = (j * 7u) % count;
      EXPECT_FLOAT_EQ(static_cast<float>(ct(i)), expect[i]) << i;
    }
    std::vector<float> all(count);
    static_cast<HoughRomTensor*>(*t)->decode_all(all.data());
    for (uint32_t i = 0; i < count; ++i) EXPECT_FLOAT_EQ(all[i], expect[i]);
  }
  EXPECT_FALSE(got_error);
}

TEST_F(HoughFixture, fully_connected_matches_decoded_float_weights) {
  const uint8_t line_bits = 2, t_bits = 6;
  std::vector<float> lines;
  make_lines(lines, t_bits);
  const uint16_t in_f = 12, out_f = 5;
  const uint32_t count = in_f * out_f;
  const uint16_t stride = out_f;  // pair neighbouring input features
  const uint32_t pairs = hough::pair_count(count, stride);
  std::vector<uint32_t> codes(pairs);
  for (uint32_t p = 0; p < pairs; ++p) codes[p] = (p * 40503u + 11u) & 255u;
  std::vector<uint8_t> stream(hough::stream_bytes(count, stride, line_bits, t_bits));
  packing::pack_bits(codes.data(), pairs, line_bits + t_bits, stream.data());
  HoughEncoding e = HoughEncoding::make(line_bits, t_bits, stride, lines.data());

  std::vector<float> wf(count), x(in_f), bias(out_f), y_h(out_f), y_f(out_f);
  hough::decode_all(stream.data(), lines.data(), line_bits, t_bits, stride, count, wf.data());
  for (int i = 0; i < in_f; ++i) x[i] = 0.1f * i - 0.4f;
  for (int o = 0; o < out_f; ++o) bias[o] = 0.01f * o;

  typedef FullyConnectedOperator<float> FC;
  {
    Tensor tx = new RomTensor({1, in_f}, flt, x.data());
    Tensor tw = new HoughRomTensor({in_f, out_f}, e, stream.data());
    Tensor tb = new RomTensor({out_f}, flt, bias.data());
    Tensor ty = new BufferTensor({1, out_f}, flt, y_h.data());
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
  for (int o = 0; o < out_f; ++o) EXPECT_NEAR(y_h[o], y_f[o], 1e-5f);
  EXPECT_FALSE(got_error);
}

TEST_F(HoughFixture, rejects_bad_encodings) {
  float lines[4] = {0, 0, 1, 1};
  uint8_t stream[4] = {0};
  Tensor a = new HoughRomTensor({2}, HoughEncoding::make(9, 1, 1, lines), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor b = new HoughRomTensor({2}, HoughEncoding::make(8, 9, 1, lines), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor c = new HoughRomTensor({2}, HoughEncoding::make(0, 4, 0, lines), stream);
  EXPECT_TRUE(got_error);
  got_error = false;
  Tensor d = new HoughRomTensor({2}, HoughEncoding::make(0, 4, 1, lines), stream);
  EXPECT_FALSE(got_error);
}
