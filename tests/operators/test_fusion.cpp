// Fused operators against the unfused operator chains they replace.
//
//  - Activation-only fusion must be bit-identical to "op, then activation".
//  - Batch-norm folding (scale input, or scale folded into the weights) must
//    match "op, then BatchNormOperator" to float rounding.
//  - The non-dense fallback (packed weights) must give the same result as
//    the dense path.
//  - int8 add must be bit-exact with TFLite's ADD; constants_int8_add.hpp
//    holds TFLite's own outputs (gen_constants_int8_add.py).
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "pTensor.h"
#if PTENSOR_WITH_QUANTIZED_OPS
#include "constants_int8_add.hpp"
#endif

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

struct FusionFixture : public ::testing::Test {
  localCircularArenaAllocator<8192> meta_allocator;
  localCircularArenaAllocator<4096, uint32_t> ram_allocator;
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

std::vector<float> rnd(size_t n, float s, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(x >> 8) / 16777216.0f - 0.5f) * 2.0f * s;
  }
  return v;
}

std::vector<float> positive(size_t n, uint32_t seed) {
  std::vector<float> v = rnd(n, 0.5f, seed);
  for (float& f : v) f = 0.6f + f;  // [0.1, 1.1]
  return v;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]) / (1.0f + std::fabs(b[i]));
    if (d > m) m = d;
  }
  return m;
}

struct BN {
  std::vector<float> mean, var, gamma, beta;
  explicit BN(uint32_t c)
      : mean(rnd(c, 0.3f, 11)), var(positive(c, 12)), gamma(rnd(c, 1.5f, 13)),
        beta(rnd(c, 0.5f, 14)) {}
};

const float kEps = 1e-3f;

void run_batch_norm(std::vector<float>& x, const BN& bn,
                    const TensorShape& shape) {
  std::vector<float> y(x.size());
  const uint32_t c = static_cast<uint32_t>(bn.mean.size());
  {
    typedef BatchNormOperator<float> Op;
    Tensor tx = new RomTensor(shape, flt, x.data());
    Tensor tm = new RomTensor({static_cast<uint16_t>(c)}, flt, bn.mean.data());
    Tensor tv = new RomTensor({static_cast<uint16_t>(c)}, flt, bn.var.data());
    Tensor tg = new RomTensor({static_cast<uint16_t>(c)}, flt, bn.gamma.data());
    Tensor tb = new RomTensor({static_cast<uint16_t>(c)}, flt, bn.beta.data());
    Tensor ty = new BufferTensor(shape, flt, y.data());
    Op op(kEps);
    op.set_inputs({{Op::in, tx}, {Op::mean, tm}, {Op::variance, tv},
                   {Op::gamma, tg}, {Op::beta, tb}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  x.swap(y);
}

void relu6(std::vector<float>& x) {
  for (float& v : x) v = v < 0 ? 0 : (v > 6 ? 6 : v);
}

}  // namespace

// ---- conv2d ---------------------------------------------------------------

TEST_F(FusionFixture, conv_relu_is_bit_identical_to_conv_then_relu) {
  const uint16_t N = 1, H = 9, W = 7, C = 5, OC = 6, K = 3;
  for (int pad = 0; pad < 2; ++pad) {
    const Padding padding = pad ? SAME : VALID;
    const uint16_t OH = pad ? H : H - K + 1, OW = pad ? W : W - K + 1;
    std::vector<float> x = rnd(N * H * W * C, 1, 1), f = rnd(OC * K * K * C, 1, 2),
                       b = rnd(OC, 1, 3);
    std::vector<float> ref(N * OH * OW * OC), out(ref.size());
    {
      typedef Conv2dOperator<float> Op;
      Tensor tx = new RomTensor({N, H, W, C}, flt, x.data());
      Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
      Tensor tb = new RomTensor({OC}, flt, b.data());
      Tensor ty = new BufferTensor({N, OH, OW, OC}, flt, ref.data());
      Op op({1, 1}, padding);
      op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, ty}})
          .eval();
    }
    for (float& v : ref) v = v < 0.0f ? 0.0f : v;
    {
      typedef FusedConv2dOperator<float> Op;
      Tensor tx = new RomTensor({N, H, W, C}, flt, x.data());
      Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
      Tensor tb = new RomTensor({OC}, flt, b.data());
      Tensor ty = new BufferTensor({N, OH, OW, OC}, flt, out.data());
      Op op(1, 1, padding, FusedActivation::relu());
      op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, ty}})
          .eval();
    }
    EXPECT_TRUE(same_bits(out, ref)) << "padding " << pad;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, conv_bn_relu6_folded_matches_unfused_chain) {
  const uint16_t N = 2, H = 8, W = 8, C = 4, OC = 8, K = 3, S = 2;
  const uint16_t OH = 4, OW = 4;  // SAME, stride 2
  std::vector<float> x = rnd(N * H * W * C, 2, 4), f = rnd(OC * K * K * C, 1, 5),
                     b = rnd(OC, 1, 6);
  BN bn(OC);

  std::vector<float> ref(N * OH * OW * OC);
  {
    typedef Conv2dOperator<float> Op;
    Tensor tx = new RomTensor({N, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({OC}, flt, b.data());
    Tensor ty = new BufferTensor({N, OH, OW, OC}, flt, ref.data());
    Op op({S, S}, SAME);
    op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  run_batch_norm(ref, bn, {N, OH, OW, OC});
  relu6(ref);

  std::vector<float> scale(OC), bias(OC);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(),
                          bn.beta.data(), kEps, b.data(), OC, scale.data(),
                          bias.data());
  std::vector<float> folded_f = f;
  fusion::fold_conv2d_weights(folded_f.data(), OC, K * K * C, scale.data());

  typedef FusedConv2dOperator<float> Op;
  // (a) scale as an input, weights untouched; (b) scale folded into weights.
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<float> out(ref.size());
    {
      Tensor tx = new RomTensor({N, H, W, C}, flt, x.data());
      Tensor tf = new RomTensor({OC, K, K, C}, flt,
                                mode ? folded_f.data() : f.data());
      Tensor tb = new RomTensor({OC}, flt, bias.data());
      Tensor ts = new RomTensor({OC}, flt, scale.data());
      Tensor ty = new BufferTensor({N, OH, OW, OC}, flt, out.data());
      Op op(S, S, SAME, FusedActivation::relu6());
      if (mode)
        op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}});
      else
        op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}, {Op::scale, ts}});
      op.set_outputs({{Op::out, ty}}).eval();
    }
    EXPECT_LT(max_rel_diff(out, ref), 2e-6f) << "mode " << mode;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, conv_packed_weights_matches_dense) {
  const uint16_t H = 6, W = 6, C = 3, OC = 4, K = 3;
  std::vector<float> x = rnd(H * W * C, 1, 7), b = rnd(OC, 1, 8),
                     s = positive(OC, 9);
  // 8-bit packed weights and the same weights decoded to float.
  const uint32_t n = OC * K * K * C;
  std::vector<uint32_t> codes(n);
  for (uint32_t i = 0; i < n; ++i) codes[i] = (i * 37u + 11u) % 256u;
  PackedEncoding enc = PackedEncoding::bits(8, 0.01f, 128);
  std::vector<uint8_t> payload(packing::packed_size_bytes(n, 8));
  packing::pack_bits(codes.data(), n, 8, payload.data());
  std::vector<float> f(n);
  for (uint32_t i = 0; i < n; ++i) f[i] = (static_cast<int32_t>(codes[i]) - 128) * 0.01f;

  std::vector<float> dense_out(H * W * OC), packed_out(H * W * OC);
  typedef FusedConv2dOperator<float> Op;
  for (int packed = 0; packed < 2; ++packed) {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = packed ? Tensor(new PackedRomTensor({OC, K, K, C}, flt, enc, payload.data()))
                       : Tensor(new RomTensor({OC, K, K, C}, flt, f.data()));
    Tensor tb = new RomTensor({OC}, flt, b.data());
    Tensor ts = new RomTensor({OC}, flt, s.data());
    Tensor ty = new BufferTensor({1, H, W, OC}, flt,
                                 packed ? packed_out.data() : dense_out.data());
    Op op(1, 1, SAME, FusedActivation::relu());
    op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}, {Op::scale, ts}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  EXPECT_TRUE(same_bits(packed_out, dense_out));
  EXPECT_EQ(errors, 0);
}

// ---- depthwise ------------------------------------------------------------

TEST_F(FusionFixture, depthwise_relu6_is_bit_identical_and_bn_folds) {
  const uint16_t H = 7, W = 7, C = 6, M = 2, OC = C * M, K = 3;
  std::vector<float> x = rnd(H * W * C, 4, 21), f = rnd(K * K * OC, 1, 22),
                     b = rnd(OC, 1, 23);
  std::vector<float> plain(H * W * OC);
  {
    typedef DepthwiseConv2dOperator<float> Op;
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({1, K, K, OC}, flt, f.data());
    Tensor tb = new RomTensor({OC}, flt, b.data());
    Tensor ty = new BufferTensor({1, H, W, OC}, flt, plain.data());
    Op op(1, 1, SAME, M);
    op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  typedef FusedDepthwiseConv2dOperator<float> Op;
  // Activation only.
  {
    std::vector<float> ref = plain, out(plain.size());
    relu6(ref);
    {
      Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
      Tensor tf = new RomTensor({1, K, K, OC}, flt, f.data());
      Tensor tb = new RomTensor({OC}, flt, b.data());
      Tensor ty = new BufferTensor({1, H, W, OC}, flt, out.data());
      Op op(1, 1, SAME, FusedActivation::relu6(), M);
      op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, ty}})
          .eval();
    }
    EXPECT_TRUE(same_bits(out, ref));
  }
  // With batch norm, folded both ways; the output tensor is a RamTensor so
  // the generic (non-dense output) path runs for one of them.
  BN bn(OC);
  std::vector<float> ref = plain;
  run_batch_norm(ref, bn, {1, H, W, OC});
  relu6(ref);
  std::vector<float> scale(OC), bias(OC);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(),
                          bn.beta.data(), kEps, b.data(), OC, scale.data(),
                          bias.data());
  std::vector<float> folded_f = f;
  fusion::fold_depthwise_weights(folded_f.data(), K * K, OC, scale.data());
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<float> out(ref.size());
    {
      Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
      Tensor tf = new RomTensor({1, K, K, OC}, flt, mode ? folded_f.data() : f.data());
      Tensor tb = new RomTensor({OC}, flt, bias.data());
      Tensor ts = new RomTensor({OC}, flt, scale.data());
      Tensor ty = new RamTensor({1, H, W, OC}, flt);
      Op op(1, 1, SAME, FusedActivation::relu6(), M);
      if (mode)
        op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}});
      else
        op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}, {Op::scale, ts}});
      op.set_outputs({{Op::out, ty}}).eval();
      for (uint32_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(ty(i));
    }
    EXPECT_LT(max_rel_diff(out, ref), 2e-6f) << "mode " << mode;
  }
  EXPECT_EQ(errors, 0);
}

// ---- fully connected ------------------------------------------------------

TEST_F(FusionFixture, fully_connected_matches_unfused) {
  const uint16_t B = 3, D = 40, U = 12;
  std::vector<float> x = rnd(B * D, 1, 31), w = rnd(D * U, 1, 32), b = rnd(U, 1, 33);
  std::vector<float> plain(B * U);
  {
    typedef FullyConnectedOperator<float> FC;
    Tensor tx = new RomTensor({B, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor ty = new BufferTensor({B, U}, flt, plain.data());
    FC op(Fuseable::ReLU<float>);
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}})
        .set_outputs({{FC::output, ty}})
        .eval();
  }
  typedef FusedFullyConnectedOperator<float> Op;
  {
    std::vector<float> out(plain.size());
    Tensor tx = new RomTensor({B, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor ty = new BufferTensor({B, U}, flt, out.data());
    Op op(FusedActivation::relu());
    op.set_inputs({{Op::input, tx}, {Op::filter, tw}, {Op::bias, tb}})
        .set_outputs({{Op::output, ty}})
        .eval();
    EXPECT_TRUE(same_bits(out, plain));
  }
  // Dense -> BN -> ReLU (BatchNorm after a ReLU-less dense layer).
  std::vector<float> lin(B * U);
  {
    typedef FullyConnectedOperator<float> FC;
    Tensor tx = new RomTensor({B, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor ty = new BufferTensor({B, U}, flt, lin.data());
    FC op;
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}})
        .set_outputs({{FC::output, ty}})
        .eval();
  }
  BN bn(U);
  run_batch_norm(lin, bn, {B, U});
  for (float& v : lin) v = v < 0 ? 0 : v;
  std::vector<float> scale(U), bias(U);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(),
                          bn.beta.data(), kEps, b.data(), U, scale.data(),
                          bias.data());
  std::vector<float> folded_w = w;
  fusion::fold_dense_weights(folded_w.data(), D, U, scale.data());
  for (int mode = 0; mode < 3; ++mode) {
    std::vector<float> out(lin.size());
    {
      // mode 2: input given as a 4D feature map (no reshape needed), and
      // the weights through a RamTensor (generic path).
      Tensor tx = mode == 2 ? Tensor(new RomTensor({B, 1, 5, 8}, flt, x.data()))
                            : Tensor(new RomTensor({B, D}, flt, x.data()));
      Tensor tw;
      if (mode == 2) {
        tw = new RamTensor({D, U}, flt);
        for (uint32_t i = 0; i < folded_w.size(); ++i) tw(i) = folded_w[i];
      } else {
        tw = new RomTensor({D, U}, flt, mode ? folded_w.data() : w.data());
      }
      Tensor tb = new RomTensor({U}, flt, bias.data());
      Tensor ts = new RomTensor({U}, flt, scale.data());
      Tensor ty = new BufferTensor({B, U}, flt, out.data());
      Op op(FusedActivation::relu());
      if (mode)
        op.set_inputs({{Op::input, tx}, {Op::filter, tw}, {Op::bias, tb}});
      else
        op.set_inputs({{Op::input, tx}, {Op::filter, tw}, {Op::bias, tb}, {Op::scale, ts}});
      op.set_outputs({{Op::output, ty}}).eval();
    }
    EXPECT_LT(max_rel_diff(out, lin), 2e-6f) << "mode " << mode;
  }
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, fully_connected_rejects_bad_shapes) {
  std::vector<float> x(8), w(12), y(3), s(4);
  typedef FusedFullyConnectedOperator<float> Op;
  Tensor tx = new RomTensor({2, 4}, flt, x.data());
  Tensor tw = new RomTensor({4, 3}, flt, w.data());
  Tensor ts = new RomTensor({4}, flt, s.data());  // should be [3]
  Tensor ty = new BufferTensor({1, 3}, flt, y.data());  // batch should be 2
  Op op;
  op.set_inputs({{Op::input, tx}, {Op::filter, tw}, {Op::scale, ts}})
      .set_outputs({{Op::output, ty}})
      .eval();
  EXPECT_EQ(errors, 1);
}

// ---- add ------------------------------------------------------------------

TEST_F(FusionFixture, add_relu_broadcast_is_bit_identical) {
  const uint16_t H = 4, W = 5, C = 6;
  std::vector<float> a = rnd(H * W * C, 1, 41), b = rnd(C, 1, 42);
  std::vector<float> ref(a.size()), out(a.size());
  {
    typedef BroadcastAddOperator<float> Op;
    Tensor ta = new RomTensor({1, H, W, C}, flt, a.data());
    Tensor tb = new RomTensor({C}, flt, b.data());
    Tensor tc = new BufferTensor({1, H, W, C}, flt, ref.data());
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  for (float& v : ref) v = v < 0 ? 0 : v;
  {
    typedef FusedAddOperator<float> Op;
    Tensor ta = new RomTensor({1, H, W, C}, flt, a.data());
    Tensor tb = new RomTensor({C}, flt, b.data());
    Tensor tc = new BufferTensor({1, H, W, C}, flt, out.data());
    Op op(FusedActivation::relu());
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  EXPECT_TRUE(same_bits(out, ref));
  EXPECT_EQ(errors, 0);
}

#if PTENSOR_WITH_QUANTIZED_OPS
// ---- int8 add -------------------------------------------------------------

namespace {
void run_int8_add(const int8_t* a, float sa, int32_t za, const int8_t* b,
                  float sb, int32_t zb, int8_t* out, float so, int32_t zo,
                  TFLM::TfLiteFusedActivation act) {
  typedef TflmSymQuantOps::QuantizedAddOperator<int8_t> Op;
  const uint16_t n = static_cast<uint16_t>(int8_add_n);
  Tensor ta = new RomTensor({n}, i8, a);
  Tensor tb = new RomTensor({n}, i8, b);
  Tensor tc = new BufferTensor({n}, i8, out);
  ta->set_quantization_params(PerTensorQuantizationParams(za, sa));
  tb->set_quantization_params(PerTensorQuantizationParams(zb, sb));
  tc->set_quantization_params(PerTensorQuantizationParams(zo, so));
  Op op(act);
  op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
}
}  // namespace

TEST_F(FusionFixture, int8_add_matches_tflite) {
  int8_t out[64] = {0};
  run_int8_add(add_none_a, add_none_a_scale, add_none_a_zp, add_none_b,
               add_none_b_scale, add_none_b_zp, out, add_none_out_scale,
               add_none_out_zp, TFLM::kTfLiteActNone);
  for (int i = 0; i < int8_add_n; ++i) EXPECT_EQ(out[i], add_none_out[i]) << i;
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, int8_add_fused_relu_matches_tflite) {
  int8_t out[64] = {0};
  run_int8_add(add_relu_a, add_relu_a_scale, add_relu_a_zp, add_relu_b,
               add_relu_b_scale, add_relu_b_zp, out, add_relu_out_scale,
               add_relu_out_zp, TFLM::kTfLiteActRelu);
  for (int i = 0; i < int8_add_n; ++i) EXPECT_EQ(out[i], add_relu_out[i]) << i;
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, int8_add_relu_clamps_at_zero_point) {
  // Same quantisation as the no-activation case, so the ReLU must clamp at
  // the output zero point: equal to "add, then clamp" bit for bit.
  int8_t plain[64], fused[64];
  run_int8_add(add_none_a, add_none_a_scale, add_none_a_zp, add_none_b,
               add_none_b_scale, add_none_b_zp, plain, add_none_out_scale,
               add_none_out_zp, TFLM::kTfLiteActNone);
  run_int8_add(add_none_a, add_none_a_scale, add_none_a_zp, add_none_b,
               add_none_b_scale, add_none_b_zp, fused, add_none_out_scale,
               add_none_out_zp, TFLM::kTfLiteActRelu);
  int clamped = 0;
  for (int i = 0; i < int8_add_n; ++i) {
    const int8_t expect = plain[i] < add_none_out_zp
                              ? static_cast<int8_t>(add_none_out_zp)
                              : plain[i];
    EXPECT_EQ(fused[i], expect) << i;
    clamped += plain[i] < add_none_out_zp;
  }
  EXPECT_GT(clamped, 0);
  EXPECT_EQ(errors, 0);
}

TEST_F(FusionFixture, int8_add_broadcast_and_packed_input) {
  // [2, 3] + [3]: the fallback path (b behind a RamTensor) must agree with
  // the element-wise formula used for the dense path.
  const int8_t a[6] = {-128, -5, 0, 17, 90, 127};
  const int8_t b[3] = {-20, 3, 100};
  int8_t dense[6], generic[6];
  typedef TflmSymQuantOps::QuantizedAddOperator<int8_t> Op;
  for (int g = 0; g < 2; ++g) {
    Tensor ta = new RomTensor({2, 3}, i8, a);
    Tensor tb;
    if (g) {
      tb = new RamTensor({3}, i8);
      for (uint32_t i = 0; i < 3; ++i) tb(i) = b[i];
    } else {
      tb = new RomTensor({3}, i8, b);
    }
    Tensor tc = new BufferTensor({2, 3}, i8, g ? generic : dense);
    ta->set_quantization_params(PerTensorQuantizationParams(-4, 0.05f));
    tb->set_quantization_params(PerTensorQuantizationParams(7, 0.11f));
    tc->set_quantization_params(PerTensorQuantizationParams(3, 0.157f));
    Op op(TFLM::kTfLiteActRelu6);
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(generic[i], dense[i]) << i;
    // Real value check, within one output step.
    const float real = (a[i] + 4) * 0.05f + (b[i % 3] - 7) * 0.11f;
    const float clamped = real < 0 ? 0 : (real > 6 ? 6 : real);
    EXPECT_NEAR((dense[i] - 3) * 0.157f, clamped, 0.157f) << i;
  }
  EXPECT_EQ(errors, 0);
}
#endif
