// Operator audit: every reference operator that the C API (and so the NumPy
// tests in python/tests) does not reach, checked against straightforward
// double-precision references with TensorFlow semantics, on both the dense
// pointer fast path and the per-element reference path.
//
// The reference path is forced with OpaqueTensor, a BufferTensor that hides
// its raw pointer (as a PackedRomTensor or a custom tensor would).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"
#include "pTensor/ops/reference/ReduceOperator.hpp"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

class OpaqueTensor : public BufferTensor {
 public:
  OpaqueTensor(const TensorShape& shape, ttype type, void* buffer)
      : BufferTensor(shape, type, buffer) {}
  virtual const void* raw_data() const override { return nullptr; }
  virtual void* raw_mutable_data() override { return nullptr; }
};

// A tensor over `buf`: dense (fast path) or opaque (reference path).
template <typename T>
TensorInterface* make(bool opaque, const TensorShape& shape, std::vector<T>& buf) {
  if (opaque)
    return new OpaqueTensor(shape, ttype_from<T>::type, buf.data());
  return new BufferTensor(shape, ttype_from<T>::type, buf.data());
}

struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * (next() & 0xFFFF) / 65535.0f;
  }
  int range(int lo, int hi) { return lo + static_cast<int>(next() % (hi - lo + 1)); }
};

std::vector<float> randf(size_t n, uint32_t seed, float lo = -2.f, float hi = 2.f) {
  Lcg r(seed);
  std::vector<float> v(n);
  for (auto& x : v) x = r.uniform(lo, hi);
  return v;
}

std::vector<int8_t> randi8(size_t n, uint32_t seed) {
  Lcg r(seed);
  std::vector<int8_t> v(n);
  for (auto& x : v) x = static_cast<int8_t>(r.range(-128, 127));
  return v;
}

// TensorFlow windowed output size and leading padding.
int out_size(int in, int k, int s, int dil, Padding p) {
  const int ek = (k - 1) * dil + 1;
  return p == SAME ? (in + s - 1) / s : (in - ek) / s + 1;
}
int pad_before(int in, int out, int k, int s, int dil, Padding p) {
  if (p != SAME) return 0;
  const int total = (out - 1) * s + (k - 1) * dil + 1 - in;
  return total > 0 ? total / 2 : 0;
}

struct AuditFixture : public ::testing::Test {
  localCircularArenaAllocator<8192> meta_allocator;
  localCircularArenaAllocator<16384, uint32_t> ram_allocator;
  SimpleErrorHandler err{10};
  int errors = 0;

  void SetUp() override {
    err.set_onError([this](Error*) { ++errors; });
    Context::get_default_context()->set_ErrorHandler(&err);
    Context::get_default_context()->set_metadata_allocator(&meta_allocator);
    Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  }
  void TearDown() override { err.set_onError([](Error*) {}); }
};

// --- pooling -----------------------------------------------------------------

enum PoolKind { kMax, kAvg, kMin };

std::vector<float> pool_ref(const std::vector<float>& x, int n, int h, int w,
                            int c, int kh, int kw, int sh, int sw, Padding p,
                            PoolKind kind, int& oh, int& ow) {
  oh = out_size(h, kh, sh, 1, p);
  ow = out_size(w, kw, sw, 1, p);
  const int pt = pad_before(h, oh, kh, sh, 1, p);
  const int pl = pad_before(w, ow, kw, sw, 1, p);
  std::vector<float> y(static_cast<size_t>(n) * oh * ow * c);
  for (int b = 0; b < n; ++b)
    for (int oy = 0; oy < oh; ++oy)
      for (int ox = 0; ox < ow; ++ox)
        for (int ch = 0; ch < c; ++ch) {
          double acc = kind == kMax ? -1e30 : (kind == kMin ? 1e30 : 0.0);
          int count = 0;
          for (int ky = 0; ky < kh; ++ky)
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - pt + ky, ix = ox * sw - pl + kx;
              if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;  // TF ignores padding
              const double v = x[((static_cast<size_t>(b) * h + iy) * w + ix) * c + ch];
              acc = kind == kMax ? std::max(acc, v)
                                 : (kind == kMin ? std::min(acc, v) : acc + v);
              ++count;
            }
          if (kind == kAvg) acc /= count;
          y[((static_cast<size_t>(b) * oh + oy) * ow + ox) * c + ch] = static_cast<float>(acc);
        }
  return y;
}

template <typename Op>
std::vector<float> run_pool(bool opaque, std::vector<float> x, int n, int h,
                            int w, int c, int kh, int kw, int sh, int sw,
                            Padding p, int oh, int ow) {
  std::vector<float> y(static_cast<size_t>(n) * oh * ow * c, 12345.f);
  {
    Tensor tx = make(opaque, {(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, x);
    Tensor ty = make(opaque, {(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)c}, y);
    Op op({(uint16_t)kh, (uint16_t)kw}, {1, (uint16_t)sh, (uint16_t)sw, 1}, p);
    op.set_inputs({{Op::in, tx}}).set_outputs({{Op::out, ty}}).eval();
  }
  return y;
}

struct PoolCase { int h, w, c, kh, kw, sh, sw; Padding p; };
const PoolCase kPoolCases[] = {
    {6, 6, 2, 2, 2, 2, 2, VALID}, {7, 5, 3, 3, 3, 2, 2, VALID},
    {7, 7, 2, 3, 3, 2, 2, SAME},  {5, 6, 2, 2, 2, 2, 2, SAME},
    {6, 6, 1, 3, 3, 1, 1, SAME},  {8, 8, 2, 1, 1, 4, 4, VALID},
    {11, 9, 2, 2, 2, 4, 4, VALID}, {8, 8, 2, 2, 2, 4, 4, SAME},
    {8, 9, 1, 1, 1, 4, 3, SAME},
};

template <typename Op>
void check_pool(PoolKind kind, int& errors) {
  int idx = 0;
  for (const PoolCase& pc : kPoolCases) {
    // All-negative inputs show a zero leaking in from the padding.
    const float lo = kind == kMax ? -3.f : -1.f, hi = kind == kMin ? 3.f : (kind == kMax ? -0.5f : 1.f);
    const auto x = randf(2u * pc.h * pc.w * pc.c, 100 + idx, kind == kMin ? 0.5f : lo, hi);
    int oh, ow;
    const auto ref = pool_ref(x, 2, pc.h, pc.w, pc.c, pc.kh, pc.kw, pc.sh, pc.sw, pc.p, kind, oh, ow);
    for (int opaque = 0; opaque < 2; ++opaque) {
      const auto got = run_pool<Op>(opaque, x, 2, pc.h, pc.w, pc.c, pc.kh, pc.kw, pc.sh, pc.sw, pc.p, oh, ow);
      for (size_t i = 0; i < ref.size(); ++i)
        ASSERT_NEAR(got[i], ref[i], 1e-5) << "case " << idx << " opaque " << opaque << " i " << i;
    }
    ++idx;
  }
  EXPECT_EQ(errors, 0);
}

}  // namespace

TEST_F(AuditFixture, maxpool_matches_tensorflow) { check_pool<MaxPoolOperator<float> >(kMax, errors); }
TEST_F(AuditFixture, avgpool_matches_tensorflow) { check_pool<AvgPoolOperator<float> >(kAvg, errors); }
TEST_F(AuditFixture, minpool_matches_tensorflow) { check_pool<MinPoolOperator<float> >(kMin, errors); }

// int8 average pool: TFLite rounds the mean half away from zero, and the sum
// must not overflow the element type.
TEST_F(AuditFixture, avgpool_int8_rounds_and_does_not_overflow) {
  const int h = 4, w = 4, c = 2;
  std::vector<int8_t> x(h * w * c);
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<int8_t>(i % 3 == 0 ? 120 : 101 - (int)i);
  for (Padding p : {VALID, SAME}) {
    const int kh = 3, kw = 3, s = 2;
    const int oh = out_size(h, kh, s, 1, p), ow = out_size(w, kw, s, 1, p);
    const int pt = pad_before(h, oh, kh, s, 1, p), pl = pad_before(w, ow, kw, s, 1, p);
    for (int opaque = 0; opaque < 2; ++opaque) {
      std::vector<int8_t> y(oh * ow * c, 0);
      {
        Tensor tx = make(opaque, {1, (uint16_t)h, (uint16_t)w, (uint16_t)c}, x);
        Tensor ty = make(opaque, {1, (uint16_t)oh, (uint16_t)ow, (uint16_t)c}, y);
        AvgPoolOperator<int8_t> op({kh, kw}, {1, s, s, 1}, p);
        op.set_inputs({{AvgPoolOperator<int8_t>::in, tx}})
            .set_outputs({{AvgPoolOperator<int8_t>::out, ty}})
            .eval();
      }
      for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox)
          for (int ch = 0; ch < c; ++ch) {
            int sum = 0, cnt = 0;
            for (int ky = 0; ky < kh; ++ky)
              for (int kx = 0; kx < kw; ++kx) {
                const int iy = oy * s - pt + ky, ix = ox * s - pl + kx;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                sum += x[(iy * w + ix) * c + ch];
                ++cnt;
              }
            const int expect = sum > 0 ? (sum + cnt / 2) / cnt : (sum - cnt / 2) / cnt;
            EXPECT_EQ(y[(oy * ow + ox) * c + ch], expect)
                << "padding " << p << " opaque " << opaque << " at " << oy << "," << ox << "," << ch;
          }
    }
  }
  EXPECT_EQ(errors, 0);
}

// --- convolution -------------------------------------------------------------

namespace {

// x NHWC, f OHWI, out NHWC.
std::vector<float> conv_ref(const std::vector<float>& x, const std::vector<float>& f,
                            const float* bias, int n, int h, int w, int c, int oc,
                            int kh, int kw, int sh, int sw, Padding p, int& oh, int& ow) {
  oh = out_size(h, kh, sh, 1, p);
  ow = out_size(w, kw, sw, 1, p);
  const int pt = pad_before(h, oh, kh, sh, 1, p), pl = pad_before(w, ow, kw, sw, 1, p);
  std::vector<float> y(static_cast<size_t>(n) * oh * ow * oc);
  for (int b = 0; b < n; ++b)
    for (int oy = 0; oy < oh; ++oy)
      for (int ox = 0; ox < ow; ++ox)
        for (int o = 0; o < oc; ++o) {
          double acc = bias ? bias[o] : 0.0;
          for (int ky = 0; ky < kh; ++ky)
            for (int kx = 0; kx < kw; ++kx) {
              const int iy = oy * sh - pt + ky, ix = ox * sw - pl + kx;
              if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
              for (int i = 0; i < c; ++i)
                acc += static_cast<double>(x[((static_cast<size_t>(b) * h + iy) * w + ix) * c + i]) *
                       f[((static_cast<size_t>(o) * kh + ky) * kw + kx) * c + i];
            }
          y[((static_cast<size_t>(b) * oh + oy) * ow + ox) * oc + o] = static_cast<float>(acc);
        }
  return y;
}

struct ConvCase { int h, w, c, oc, kh, kw, sh, sw; Padding p; };
const ConvCase kConvCases[] = {
    {7, 6, 3, 4, 3, 3, 1, 1, SAME},  {7, 6, 3, 4, 3, 3, 2, 2, SAME},
    {7, 6, 3, 4, 3, 3, 2, 2, VALID}, {8, 8, 2, 3, 1, 1, 4, 4, SAME},
    {8, 8, 2, 3, 2, 2, 4, 4, SAME},  {11, 11, 2, 3, 2, 2, 4, 4, VALID},
    {9, 7, 2, 2, 2, 3, 3, 2, VALID}, {5, 5, 1, 2, 5, 5, 1, 1, SAME},
};

}  // namespace

TEST_F(AuditFixture, conv2d_float_matches_tensorflow) {
  int idx = 0;
  for (const ConvCase& cc : kConvCases) {
    const int n = 2;
    auto x = randf(static_cast<size_t>(n) * cc.h * cc.w * cc.c, 200 + idx);
    auto f = randf(static_cast<size_t>(cc.oc) * cc.kh * cc.kw * cc.c, 300 + idx);
    auto b = randf(cc.oc, 400 + idx);
    int oh, ow;
    for (int with_bias = 0; with_bias < 2; ++with_bias) {
      const auto ref = conv_ref(x, f, with_bias ? b.data() : nullptr, n, cc.h, cc.w,
                                cc.c, cc.oc, cc.kh, cc.kw, cc.sh, cc.sw, cc.p, oh, ow);
      for (int opaque = 0; opaque < 2; ++opaque) {
        std::vector<float> y(ref.size(), 999.f);
        {
          Tensor tx = make(opaque, {(uint16_t)n, (uint16_t)cc.h, (uint16_t)cc.w, (uint16_t)cc.c}, x);
          Tensor tf = make(opaque, {(uint16_t)cc.oc, (uint16_t)cc.kh, (uint16_t)cc.kw, (uint16_t)cc.c}, f);
          Tensor tb = make(opaque, {(uint16_t)cc.oc}, b);
          Tensor ty = make(opaque, {(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)cc.oc}, y);
          Conv2dOperator<float> op({(uint16_t)cc.sh, (uint16_t)cc.sw}, cc.p);
          if (with_bias)
            op.set_inputs({{Conv2dOperator<float>::in, tx}, {Conv2dOperator<float>::filter, tf},
                           {Conv2dOperator<float>::bias, tb}});
          else
            op.set_inputs({{Conv2dOperator<float>::in, tx}, {Conv2dOperator<float>::filter, tf}});
          op.set_outputs({{Conv2dOperator<float>::out, ty}}).eval();
        }
        for (size_t i = 0; i < ref.size(); ++i)
          ASSERT_NEAR(y[i], ref[i], 1e-4) << "case " << idx << " bias " << with_bias
                                          << " opaque " << opaque << " i " << i;
      }
    }
    ++idx;
  }
  EXPECT_EQ(errors, 0);
}

// The float-emulated int8 Conv2dOperator<int8_t> (reference ops) must round
// to nearest, like dequantize -> float conv -> quantize.
TEST_F(AuditFixture, conv2d_int8_reference_rounds_to_nearest) {
  const int h = 6, w = 5, c = 3, oc = 4, k = 3;
  auto x = randi8(h * w * c, 11);
  auto f = randi8(oc * k * k * c, 12);
  std::vector<int32_t> b(oc);
  for (int o = 0; o < oc; ++o) b[o] = 300 * o - 500;
  const float xs = 0.05f, fs = 0.02f, bs = xs * fs, ys = 0.1f;
  const int32_t xz = 3, yz = -7;
  std::vector<float> xf(x.size()), ff(f.size()), bf(oc);
  for (size_t i = 0; i < x.size(); ++i) xf[i] = (x[i] - xz) * xs;
  for (size_t i = 0; i < f.size(); ++i) ff[i] = f[i] * fs;
  for (int o = 0; o < oc; ++o) bf[o] = b[o] * bs;
  for (Padding p : {SAME, VALID}) {
    int oh, ow;
    const auto ref = conv_ref(xf, ff, bf.data(), 1, h, w, c, oc, k, k, 1, 1, p, oh, ow);
    std::vector<int8_t> y(ref.size(), 0);
    {
      Tensor tx = new RomTensor({1, h, w, c}, i8, x.data());
      tx->set_quantization_params(PerTensorQuantizationParams(xz, xs));
      Tensor tf = new RomTensor({oc, k, k, c}, i8, f.data());
      tf->set_quantization_params(PerTensorQuantizationParams(0, fs));
      Tensor tb = new RomTensor({oc}, i32, b.data());
      tb->set_quantization_params(PerTensorQuantizationParams(0, bs));
      Tensor ty = new BufferTensor({1, (uint16_t)oh, (uint16_t)ow, oc}, i8, y.data());
      ty->set_quantization_params(PerTensorQuantizationParams(yz, ys));
      Conv2dOperator<int8_t> op({1, 1}, p);
      op.set_inputs({{Conv2dOperator<int8_t>::in, tx}, {Conv2dOperator<int8_t>::filter, tf},
                     {Conv2dOperator<int8_t>::bias, tb}})
          .set_outputs({{Conv2dOperator<int8_t>::out, ty}})
          .eval();
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      const long q = std::lround(ref[i] / ys) + yz;
      const int expect = static_cast<int>(std::min(127L, std::max(-128L, q)));
      ASSERT_EQ(y[i], expect) << "padding " << p << " i " << i;
    }
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, depthwise_separable_conv_float_matches_two_step) {
  // depthwise [kh, kw, C, M] then pointwise [1, 1, C*M, OC]
  const int h = 6, w = 6, c = 2, m = 2, oc = 3, k = 3;
  auto x = randf(h * w * c, 21);
  auto dw = randf(k * k * c * m, 22);
  auto pw = randf(c * m * oc, 23);
  for (Padding p : {SAME, VALID}) {
    for (int s : {1, 2}) {
      const int oh = out_size(h, k, s, 1, p), ow = out_size(w, k, s, 1, p);
      const int pt = pad_before(h, oh, k, s, 1, p), pl = pad_before(w, ow, k, s, 1, p);
      std::vector<float> y(oh * ow * oc, 0.f);
      {
        Tensor tx = new RomTensor({1, h, w, c}, flt, x.data());
        Tensor td = new RomTensor({k, k, c, m}, flt, dw.data());
        Tensor tp = new RomTensor({1, 1, c * m, oc}, flt, pw.data());
        Tensor ty = new BufferTensor({1, (uint16_t)oh, (uint16_t)ow, oc}, flt, y.data());
        typedef DepthwiseSeparableConvOperator<float> Op;
        Op op({1, (uint16_t)s, (uint16_t)s, 1}, p);
        op.set_inputs({{Op::in, tx}, {Op::depthwise_filter, td}, {Op::pointwise_filter, tp}})
            .set_outputs({{Op::out, ty}})
            .eval();
      }
      for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox)
          for (int o = 0; o < oc; ++o) {
            double acc = 0;
            for (int ic = 0; ic < c; ++ic)
              for (int r = 0; r < m; ++r) {
                double d = 0;
                for (int ky = 0; ky < k; ++ky)
                  for (int kx = 0; kx < k; ++kx) {
                    const int iy = oy * s - pt + ky, ix = ox * s - pl + kx;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                    d += x[(iy * w + ix) * c + ic] * dw[((ky * k + kx) * c + ic) * m + r];
                  }
                acc += d * pw[(ic * m + r) * oc + o];
              }
            ASSERT_NEAR(y[(oy * ow + ox) * oc + o], acc, 1e-4)
                << "padding " << p << " stride " << s;
          }
    }
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, matmul_v2_bias_and_activation) {
  const int m = 3, k = 5, n = 4;
  auto a = randf(m * k, 31), b = randf(k * n, 32), bias = randf(n, 33);
  for (int opaque = 0; opaque < 2; ++opaque) {
    std::vector<float> y(m * n, 0.f);
    {
      Tensor ta = make(opaque, {m, k}, a);
      Tensor tb = make(opaque, {k, n}, b);
      Tensor tc = make(opaque, {n}, bias);
      Tensor ty = make(opaque, {m, n}, y);
      typedef MatrixMultOperatorV2<float> Op;
      Op op(Fuseable::ReLU<float>);
      op.set_inputs({{Op::input, ta}, {Op::filter, tb}, {Op::bias, tc}})
          .set_outputs({{Op::output, ty}})
          .eval();
    }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double acc = bias[j];
        for (int d = 0; d < k; ++d) acc += a[i * k + d] * b[d * n + j];
        EXPECT_NEAR(y[i * n + j], std::max(0.0, acc), 1e-5) << opaque;
      }
  }
  EXPECT_EQ(errors, 0);
}

// --- activations ---------------------------------------------------------------

namespace {
void softmax_rows(const std::vector<float>& x, size_t depth, std::vector<double>& y) {
  y.resize(x.size());
  for (size_t r = 0; r < x.size() / depth; ++r) {
    double mx = -1e30, sum = 0;
    for (size_t j = 0; j < depth; ++j) mx = std::max(mx, (double)x[r * depth + j]);
    for (size_t j = 0; j < depth; ++j) sum += std::exp(x[r * depth + j] - mx);
    for (size_t j = 0; j < depth; ++j) y[r * depth + j] = std::exp(x[r * depth + j] - mx) / sum;
  }
}
}  // namespace

// Softmax runs over the last axis whatever the rank (it used to index rank 3
// and 4 tensors as if they were 2D).
TEST_F(AuditFixture, softmax_float_every_rank) {
  const TensorShape shapes[] = {TensorShape(7), TensorShape(3, 7), TensorShape(2, 3, 7),
                                TensorShape(2, 2, 3, 7), TensorShape(1, 1, 1, 10)};
  int idx = 0;
  for (const TensorShape& s : shapes) {
    const uint32_t n = s.get_linear_size();
    const uint16_t depth = s[s.num_dims() - 1];
    auto x = randf(n, 40 + idx, -4.f, 4.f);
    std::vector<double> ref;
    softmax_rows(x, depth, ref);
    std::vector<float> y(n, 0.f);
    {
      Tensor tx = new RomTensor(s, flt, x.data());
      Tensor ty = new BufferTensor(s, flt, y.data());
      SoftmaxOperator<float> op;
      op.set_inputs({{SoftmaxOperator<float>::in, tx}})
          .set_outputs({{SoftmaxOperator<float>::out, ty}})
          .eval();
    }
    for (uint32_t i = 0; i < n; ++i) ASSERT_NEAR(y[i], ref[i], 1e-6) << "rank " << (int)s.num_dims() << " i " << i;
    std::vector<float> z = x;
    {
      Tensor tz = new BufferTensor(s, flt, z.data());
      InPlaceSoftmax<float> op;
      op.set_inputs({{InPlaceSoftmax<float>::x, tz}}).eval();
    }
    for (uint32_t i = 0; i < n; ++i) ASSERT_NEAR(z[i], ref[i], 1e-6) << "in place rank " << (int)s.num_dims();
    ++idx;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, softmax_int8_rank4) {
  const TensorShape s(1, 2, 2, 6);
  auto x = randi8(24, 50);
  const float xs = 0.1f, ys = 1.f / 256;
  const int32_t xz = 5, yz = -128;
  std::vector<float> xf(24);
  for (int i = 0; i < 24; ++i) xf[i] = (x[i] - xz) * xs;
  std::vector<double> ref;
  softmax_rows(xf, 6, ref);
  std::vector<int8_t> y(24, 0);
  {
    Tensor tx = new RomTensor(s, i8, x.data());
    tx->set_quantization_params(PerTensorQuantizationParams(xz, xs));
    Tensor ty = new BufferTensor(s, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(yz, ys));
    SoftmaxOperator<int8_t> op;
    op.set_inputs({{SoftmaxOperator<int8_t>::in, tx}})
        .set_outputs({{SoftmaxOperator<int8_t>::out, ty}})
        .eval();
  }
  for (int i = 0; i < 24; ++i) {
    const long q = std::min(127L, std::lround(ref[i] / ys) + yz);
    EXPECT_NEAR(y[i], q, 1) << i;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, logistic_float_and_in_place) {
  auto x = randf(32, 60, -8.f, 8.f);
  std::vector<float> y(32, 0.f), z = x;
  {
    Tensor tx = new RomTensor({32}, flt, x.data());
    Tensor ty = new BufferTensor({32}, flt, y.data());
    LogisticOperator<float> op;
    op.set_inputs({{LogisticOperator<float>::in, tx}})
        .set_outputs({{LogisticOperator<float>::out, ty}})
        .eval();
    Tensor tz = new BufferTensor({32}, flt, z.data());
    InPlaceLogistic<float> ip;
    ip.set_inputs({{InPlaceLogistic<float>::x, tz}}).eval();
  }
  for (int i = 0; i < 32; ++i) {
    const double s = 1.0 / (1.0 + std::exp(-(double)x[i]));
    EXPECT_NEAR(y[i], s, 1e-6) << i;
    EXPECT_NEAR(z[i], s, 1e-6) << "in place " << i;  // it used to run softmax
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, relu_relu6_float_in_place) {
  auto x = randf(40, 61, -8.f, 9.f);
  std::vector<float> a = x, b = x;
  {
    Tensor ta = new BufferTensor({40}, flt, a.data());
    Tensor tb = new BufferTensor({40}, flt, b.data());
    InPlaceReLU<float> r;
    r.set_inputs({{InPlaceReLU<float>::x, ta}}).eval();
    InPlaceReLU6<float> r6;
    r6.set_inputs({{InPlaceReLU6<float>::x, tb}}).eval();
  }
  for (int i = 0; i < 40; ++i) {
    EXPECT_EQ(a[i], std::max(0.f, x[i]));
    EXPECT_EQ(b[i], std::min(6.f, std::max(0.f, x[i])));
  }
}

TEST_F(AuditFixture, tanh_float) {
  auto x = randf(32, 62, -5.f, 5.f);
  std::vector<float> y(32, 0.f);
  {
    Tensor tx = new RomTensor({32}, flt, x.data());
    Tensor ty = new BufferTensor({32}, flt, y.data());
    TanhOperator<float> op;
    op.set_inputs({{TanhOperator<float>::act_in, tx}})
        .set_outputs({{TanhOperator<float>::act_out, ty}})
        .eval();
  }
  for (int i = 0; i < 32; ++i) EXPECT_NEAR(y[i], std::tanh((double)x[i]), 1e-6);
}

// --- shape and index ops -----------------------------------------------------

namespace {
template <typename T>
std::vector<T> transpose_ref(const std::vector<T>& x, const std::vector<int>& shape,
                             const std::vector<int>& perm, std::vector<int>& oshape) {
  const int r = static_cast<int>(shape.size());
  oshape.assign(r, 0);
  for (int i = 0; i < r; ++i) oshape[i] = shape[perm[i]];  // NumPy / TFLite
  std::vector<int> istr(r, 1);
  for (int i = r - 2; i >= 0; --i) istr[i] = istr[i + 1] * shape[i + 1];
  std::vector<T> y(x.size());
  std::vector<int> idx(r, 0);
  for (size_t o = 0; o < y.size(); ++o) {
    size_t src = 0;
    for (int i = 0; i < r; ++i) src += static_cast<size_t>(idx[i]) * istr[perm[i]];
    y[o] = x[src];
    for (int i = r - 1; i >= 0; --i) {
      if (++idx[i] < oshape[i]) break;
      idx[i] = 0;
    }
  }
  return y;
}

TensorShape to_shape(const std::vector<int>& s) {
  switch (s.size()) {
    case 1: return TensorShape((uint16_t)s[0]);
    case 2: return TensorShape((uint16_t)s[0], (uint16_t)s[1]);
    case 3: return TensorShape((uint16_t)s[0], (uint16_t)s[1], (uint16_t)s[2]);
    default: return TensorShape((uint16_t)s[0], (uint16_t)s[1], (uint16_t)s[2], (uint16_t)s[3]);
  }
}
}  // namespace

TEST_F(AuditFixture, transpose_matches_numpy_for_every_permutation) {
  const std::vector<std::vector<int> > shapes = {{2, 3}, {2, 3, 4}, {2, 3, 4, 5}};
  for (const auto& shape : shapes) {
    std::vector<int> perm(shape.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int>(i);
    size_t n = 1;
    for (int d : shape) n *= d;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<float>(i);
    do {
      std::vector<int> oshape;
      const auto ref = transpose_ref(x, shape, perm, oshape);
      std::vector<int32_t> p32(perm.begin(), perm.end());
      std::vector<float> y(n, -1.f);
      TensorShape got_shape(0, 0, 0, 0);
      {
        Tensor tx = new RomTensor(to_shape(shape), flt, x.data());
        Tensor tp = new RomTensor({(uint16_t)perm.size()}, i32, p32.data());
        Tensor ty = new BufferTensor(to_shape(oshape), flt, y.data());
        TransposeOperator<float> op;
        op.set_inputs({{TransposeOperator<float>::input, tx}, {TransposeOperator<float>::perm, tp}})
            .set_outputs({{TransposeOperator<float>::output, ty}})
            .eval();
        got_shape = ty->get_shape();
      }
      std::string ps;
      for (int v : perm) ps += std::to_string(v);
      for (size_t i = 0; i < oshape.size(); ++i) ASSERT_EQ(got_shape[i], oshape[i]) << "perm " << ps;
      for (size_t i = 0; i < n; ++i) ASSERT_EQ(y[i], ref[i]) << "perm " << ps << " i " << i;
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
  EXPECT_EQ(errors, 0);
}

// Ties go to the first index, like NumPy / TFLite.
TEST_F(AuditFixture, argmax_argmin_ties_and_axes) {
  const int d0 = 3, d1 = 4, d2 = 5;
  std::vector<float> x(d0 * d1 * d2);
  Lcg r(70);
  for (auto& v : x) v = static_cast<float>(r.range(-2, 2));  // many ties
  for (uint32_t axis = 0; axis < 3; ++axis) {
    const int dims[3] = {d0, d1, d2};
    int osz = 1;
    for (int i = 0; i < 3; ++i) if (i != (int)axis) osz *= dims[i];
    for (int is_min = 0; is_min < 2; ++is_min) {
      std::vector<uint32_t> y(osz, 99u), ax(1, axis);
      {
        Tensor tx = new RomTensor({d0, d1, d2}, flt, x.data());
        Tensor ta = new RomTensor({1}, u32, ax.data());
        Tensor ty = new BufferTensor({(uint16_t)osz}, u32, y.data());
        if (is_min) {
          ArgMinOperator<float> op;
          op.set_inputs({{ArgMinOperator<float>::input, tx}, {ArgMinOperator<float>::axis, ta}})
              .set_outputs({{ArgMinOperator<float>::output, ty}})
              .eval();
        } else {
          ArgMaxOperator<float> op;
          op.set_inputs({{ArgMaxOperator<float>::input, tx}, {ArgMaxOperator<float>::axis, ta}})
              .set_outputs({{ArgMaxOperator<float>::output, ty}})
              .eval();
        }
      }
      int o = 0;
      for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
          for (int k = 0; k < dims[2]; ++k) {
            const int idx[3] = {i, j, k};
            if (idx[axis] != 0) continue;
            int best = 0;
            float bv = 0;
            for (int t = 0; t < dims[axis]; ++t) {
              int id[3] = {i, j, k};
              id[axis] = t;
              const float v = x[(id[0] * d1 + id[1]) * d2 + id[2]];
              if (t == 0 || (is_min ? v < bv : v > bv)) { bv = v; best = t; }
            }
            // output index for (i, j, k) with the axis removed
            int oi = 0;
            for (int q = 0; q < 3; ++q) if (q != (int)axis) oi = oi * dims[q] + idx[q];
            ASSERT_EQ(y[oi], (uint32_t)best) << "axis " << axis << " min " << is_min;
            ++o;
          }
    }
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, reduce_max_min) {
  auto x = randf(50, 80, -3.f, 3.f);
  float mx = 0, mn = 0;
  {
    Tensor tx = new RomTensor({5, 10}, flt, x.data());
    Tensor tmx = new BufferTensor({1}, flt, &mx);
    Tensor tmn = new BufferTensor({1}, flt, &mn);
    MaxOperator<float> a;
    a.set_inputs({{MaxOperator<float>::in, tx}}).set_outputs({{MaxOperator<float>::out, tmx}}).eval();
    MinOperator<float> b;
    b.set_inputs({{MinOperator<float>::in, tx}}).set_outputs({{MinOperator<float>::out, tmn}}).eval();
  }
  EXPECT_EQ(mx, *std::max_element(x.begin(), x.end()));
  EXPECT_EQ(mn, *std::min_element(x.begin(), x.end()));
}

TEST_F(AuditFixture, reduce_mean_matches_numpy) {
  const int d[4] = {2, 3, 4, 5};
  auto x = randf(120, 81);
  // every non-empty subset of the axes
  for (int mask = 1; mask < 16; ++mask) {
    std::vector<uint16_t> axes;
    for (int i = 0; i < 4; ++i) if (mask & (1 << i)) axes.push_back(i);
    int osz = 1;
    for (int i = 0; i < 4; ++i) if (!(mask & (1 << i))) osz *= d[i];
    std::vector<float> y(osz, -9.f);
    {
      Tensor tx = new RomTensor({2, 3, 4, 5}, flt, x.data());
      Tensor ty = new BufferTensor({(uint16_t)osz}, flt, y.data());
      ReduceMeanOperator<float> op(axes.data(), static_cast<uint8_t>(axes.size()));
      op.set_inputs({{ReduceMeanOperator<float>::input, tx}})
          .set_outputs({{ReduceMeanOperator<float>::output, ty}})
          .eval();
    }
    std::vector<double> ref(osz, 0.0);
    int cnt = 120 / osz;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 4; ++c)
          for (int e = 0; e < 5; ++e) {
            const int id[4] = {a, b, c, e};
            int oi = 0;
            for (int q = 0; q < 4; ++q) if (!(mask & (1 << q))) oi = oi * d[q] + id[q];
            ref[oi] += x[((a * 3 + b) * 4 + c) * 5 + e];
          }
    for (int i = 0; i < osz; ++i) ASSERT_NEAR(y[i], ref[i] / cnt, 1e-5) << "mask " << mask;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, reshape_copies_in_order) {
  auto x = randf(24, 90);
  std::vector<float> y(24, 0.f);
  {
    Tensor tx = new RomTensor({2, 3, 4}, flt, x.data());
    Tensor ty = new BufferTensor({4, 6}, flt, y.data());
    ReshapeOperator<float> op(TensorShape(4, 6));
    op.set_inputs({{ReshapeOperator<float>::input, tx}})
        .set_outputs({{ReshapeOperator<float>::output, ty}})
        .eval();
  }
  EXPECT_EQ(y, x);
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, squeeze_updates_rank_and_keeps_large_dims) {
  std::vector<int8_t> buf(300 * 2, 0);
  {
    Tensor t = new BufferTensor({1, 300, 1, 2}, i8, buf.data());
    SqueezeOperator<int8_t> op;
    op.set_inputs({{SqueezeOperator<int8_t>::x, t}}).eval();
    const TensorShape& s = t->get_shape();
    EXPECT_EQ(s.num_dims(), 2);
    EXPECT_EQ(s[0], 300);
    EXPECT_EQ(s[1], 2);
    EXPECT_EQ(s.get_linear_size(), 600u);
  }
  {
    Tensor t = new BufferTensor({1, 300, 1, 2}, i8, buf.data());
    SqueezeOperator<int8_t> op({2});  // only axis 2
    op.set_inputs({{SqueezeOperator<int8_t>::x, t}}).eval();
    const TensorShape& s = t->get_shape();
    EXPECT_EQ(s.num_dims(), 3);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[1], 300);
    EXPECT_EQ(s[2], 2);
  }
  EXPECT_EQ(errors, 0);
}

// --- elementwise, reference path ------------------------------------------------

TEST_F(AuditFixture, elementwise_reference_path_broadcasts) {
  auto a = randf(2 * 3 * 4, 95), b = randf(3 * 1, 96, 0.5f, 2.f);
  for (int opaque = 0; opaque < 2; ++opaque) {
    std::vector<float> y(24, 0.f), z(24, 0.f);
    {
      Tensor ta = make(opaque, {2, 3, 4}, a);
      Tensor tb = make(opaque, {3, 1}, b);
      Tensor ty = make(opaque, {2, 3, 4}, y);
      DivOperator<float> op;
      op.set_inputs({{DivOperator<float>::a, ta}, {DivOperator<float>::b, tb}})
          .set_outputs({{DivOperator<float>::c, ty}})
          .eval();
      Tensor tz = make(opaque, {2, 3, 4}, z);
      AddOperator<float> add;
      add.set_inputs({{AddOperator<float>::a, ta}, {AddOperator<float>::b, ta}})
          .set_outputs({{AddOperator<float>::c, tz}})
          .eval();
    }
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k) {
          const int o = (i * 3 + j) * 4 + k;
          EXPECT_FLOAT_EQ(y[o], a[o] / b[j]) << opaque;
          EXPECT_FLOAT_EQ(z[o], a[o] + a[o]) << opaque;
        }
    std::vector<float> u(24, 0.f);
    {
      Tensor ta = make(opaque, {24}, a);
      Tensor tu = make(opaque, {24}, u);
      GeluOperator op;
      op.set_inputs({{GeluOperator::in, ta}}).set_outputs({{GeluOperator::out, tu}}).eval();
    }
    for (int i = 0; i < 24; ++i) {
      const double v = a[i];
      EXPECT_NEAR(u[i], 0.5 * v * (1 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))), 1e-6);
    }
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(AuditFixture, layer_ops_reference_path_matches_fast_path) {
  // depthwise conv (dilated, multiplier 2) and LSTM / GRU: opaque vs dense.
  const int h = 7, w = 6, c = 2, m = 2, k = 3;
  auto x = randf(h * w * c, 97), f = randf(k * k * c * m, 98), b = randf(c * m, 99);
  for (Padding p : {SAME, VALID}) {
    const int oh = out_size(h, k, 1, 2, p), ow = out_size(w, k, 1, 2, p);
    std::vector<float> outs[2];
    for (int opaque = 0; opaque < 2; ++opaque) {
      outs[opaque].assign(oh * ow * c * m, 0.f);
      Tensor tx = make(opaque, {1, h, w, c}, x);
      Tensor tf = make(opaque, {1, k, k, c * m}, f);
      Tensor tb = make(opaque, {c * m}, b);
      Tensor ty = make(opaque, {1, (uint16_t)oh, (uint16_t)ow, c * m}, outs[opaque]);
      typedef DepthwiseConv2dOperator<float> Op;
      Op op(1, 1, p, m, 2, 2);
      op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, ty}})
          .eval();
    }
    for (size_t i = 0; i < outs[0].size(); ++i) ASSERT_NEAR(outs[0][i], outs[1][i], 1e-5) << p;
  }
  EXPECT_EQ(errors, 0);
}
