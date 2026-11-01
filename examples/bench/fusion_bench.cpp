// Fused vs unfused operator chains. Build with -DPTENSOR_BENCH=ON and run
// build/examples/bench/ptensor_fusion_bench.
//
// For each pattern it times
//   unfused  the pTensor operators one after another (what a model had to do
//            before fusion: e.g. Conv2d -> BatchNorm -> Clip),
//   unfused* the same chain with the extra passes written as tight dense
//            loops over precomputed scale/shift, i.e. the best an unfused
//            pipeline could do; the gap to "fused" is the pure fusion gain,
//   fused    the fused operator with the batch norm passed as a scale input,
//   folded   the fused operator with the batch norm folded into the weights,
// and checks the fused outputs against the unfused ones (max |diff|, and
// whether they are bit-identical where they should be).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

localCircularArenaAllocator<16384, uint32_t, 64> meta;
localCircularArenaAllocator<4096, uint32_t, 16> ram;
SimpleErrorHandler err(4);

std::vector<float> rnd(size_t n, float s, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t x = seed * 2654435761u + 12345u;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(x >> 8) / 16777216.0f - 0.5f) * 2.0f * s;
  }
  return v;
}

struct BN {
  std::vector<float> mean, var, gamma, beta, scale, shift;
  explicit BN(uint32_t c)
      : mean(rnd(c, 0.3f, 1)), var(rnd(c, 0.4f, 2)), gamma(rnd(c, 1.0f, 3)),
        beta(rnd(c, 0.3f, 4)), scale(c), shift(c) {
    for (float& v : var) v += 0.6f;
  }
};

const float kEps = 1e-3f;

template <typename F>
double best_us(F fn) {
  double best = 1e30;
  int reps = 0;
  auto start = std::chrono::steady_clock::now();
  do {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (us < best) best = us;
    ++reps;
  } while (reps < 3 ||
           (reps < 300 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.4));
  return best;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void header(const char* name) { std::printf("\n%s\n", name); }

void row(const char* variant, double us, double base_us, const char* check) {
  std::printf("  %-10s %10.1f us   %5.2fx   %s\n", variant, us, base_us / us, check);
}

void batch_norm_op(Tensor& in, Tensor& out, BN& bn, uint16_t c) {
  typedef BatchNormOperator<float> Op;
  Tensor tm = new RomTensor({c}, flt, bn.mean.data());
  Tensor tv = new RomTensor({c}, flt, bn.var.data());
  Tensor tg = new RomTensor({c}, flt, bn.gamma.data());
  Tensor tb = new RomTensor({c}, flt, bn.beta.data());
  Op op(kEps);
  op.set_inputs({{Op::in, in}, {Op::mean, tm}, {Op::variance, tv},
                 {Op::gamma, tg}, {Op::beta, tb}})
      .set_outputs({{Op::out, out}})
      .eval();
}

void clip_op(Tensor& in, Tensor& out, float lo, float hi) {
  ClipOperator op(activations::Clip(lo, hi));
  op.set_inputs({{ClipOperator::in, in}}).set_outputs({{ClipOperator::out, out}}).eval();
}

// Tight separate passes: y = x * s + t, then clamp (in place).
void affine_then_clamp(float* y, uint32_t n, uint32_t c, const float* s,
                       const float* t, float lo, float hi) {
  for (uint32_t i = 0; i < n; i += c)
    for (uint32_t k = 0; k < c; ++k) y[i + k] = y[i + k] * s[k] + t[k];
  for (uint32_t i = 0; i < n; ++i) y[i] = y[i] < lo ? lo : (y[i] > hi ? hi : y[i]);
}

char g_check[128];
const char* diff_str(const std::vector<float>& got, const std::vector<float>& ref,
                     bool expect_identical) {
  if (same_bits(got, ref))
    std::snprintf(g_check, sizeof g_check, "bit-identical");
  else
    std::snprintf(g_check, sizeof g_check, "max |diff| %.2e%s",
                  max_abs_diff(got, ref), expect_identical ? "  (EXPECTED IDENTICAL)" : "");
  return g_check;
}

// ---------------------------------------------------------------------------

void conv_bn_relu6() {
  const uint16_t H = 32, W = 32, C = 16, OC = 32, K = 3;
  header("conv2d 1x32x32x16 3x3 -> 32 SAME, + batch norm + ReLU6");
  std::vector<float> x = rnd(H * W * C, 1, 10), f = rnd(OC * K * K * C, 0.3f, 11),
                     b = rnd(OC, 0.3f, 12);
  BN bn(OC);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(), bn.beta.data(),
                          kEps, b.data(), OC, bn.scale.data(), bn.shift.data());
  std::vector<float> ff = f;
  fusion::fold_conv2d_weights(ff.data(), OC, K * K * C, bn.scale.data());
  const uint32_t n = H * W * OC;
  std::vector<float> t1(n), t2(n), ref(n), star(n), fused(n), folded(n);

  const double base = best_us([&] {
    typedef Conv2dOperator<float> Op;
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({OC}, flt, b.data());
    Tensor y1 = new BufferTensor({1, H, W, OC}, flt, t1.data());
    Tensor y2 = new BufferTensor({1, H, W, OC}, flt, t2.data());
    Tensor y3 = new BufferTensor({1, H, W, OC}, flt, ref.data());
    Op op({1, 1}, SAME);
    op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
        .set_outputs({{Op::out, y1}}).eval();
    batch_norm_op(y1, y2, bn, OC);
    clip_op(y2, y3, 0, 6);
  });
  std::vector<float> zero(OC, 0.0f);
  const double t_star = best_us([&] {
    typedef Conv2dOperator<float> Op;
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({OC}, flt, zero.data());
    Tensor y1 = new BufferTensor({1, H, W, OC}, flt, star.data());
    Op op({1, 1}, SAME);
    op.set_inputs({{Op::in, tx}, {Op::filter, tf}, {Op::bias, tb}})
        .set_outputs({{Op::out, y1}}).eval();
    affine_then_clamp(star.data(), n, OC, bn.scale.data(), bn.shift.data(), 0, 6);
  });
  typedef FusedConv2dOperator<float> F;
  const double t_fused = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({OC, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({OC}, flt, bn.shift.data());
    Tensor ts = new RomTensor({OC}, flt, bn.scale.data());
    Tensor ty = new BufferTensor({1, H, W, OC}, flt, fused.data());
    F op(1, 1, SAME, FusedActivation::relu6());
    op.set_inputs({{F::in, tx}, {F::filter, tf}, {F::bias, tb}, {F::scale, ts}})
        .set_outputs({{F::out, ty}}).eval();
  });
  const double t_folded = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({OC, K, K, C}, flt, ff.data());
    Tensor tb = new RomTensor({OC}, flt, bn.shift.data());
    Tensor ty = new BufferTensor({1, H, W, OC}, flt, folded.data());
    F op(1, 1, SAME, FusedActivation::relu6());
    op.set_inputs({{F::in, tx}, {F::filter, tf}, {F::bias, tb}})
        .set_outputs({{F::out, ty}}).eval();
  });
  row("unfused", base, base, "3 ops, 2 intermediate tensors (256 KB)");
  row("unfused*", t_star, base, diff_str(star, ref, false));
  row("fused", t_fused, base, diff_str(fused, ref, false));
  row("folded", t_folded, base, diff_str(folded, ref, false));
}

void depthwise_bn_relu6() {
  const uint16_t H = 32, W = 32, C = 64, K = 3;
  header("depthwise 1x32x32x64 3x3 SAME, + batch norm + ReLU6");
  std::vector<float> x = rnd(H * W * C, 1, 20), f = rnd(K * K * C, 0.5f, 21),
                     b = rnd(C, 0.3f, 22);
  BN bn(C);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(), bn.beta.data(),
                          kEps, b.data(), C, bn.scale.data(), bn.shift.data());
  std::vector<float> ff = f;
  fusion::fold_depthwise_weights(ff.data(), K * K, C, bn.scale.data());
  const uint32_t n = H * W * C;
  std::vector<float> t1(n), t2(n), ref(n), star(n), fused(n), folded(n), zero(C, 0.0f);
  typedef DepthwiseConv2dOperator<float> D;
  const double base = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({1, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({C}, flt, b.data());
    Tensor y1 = new BufferTensor({1, H, W, C}, flt, t1.data());
    Tensor y2 = new BufferTensor({1, H, W, C}, flt, t2.data());
    Tensor y3 = new BufferTensor({1, H, W, C}, flt, ref.data());
    D op(1, 1, SAME, 1);
    op.set_inputs({{D::in, tx}, {D::filter, tf}, {D::bias, tb}}).set_outputs({{D::out, y1}}).eval();
    batch_norm_op(y1, y2, bn, C);
    clip_op(y2, y3, 0, 6);
  });
  const double t_star = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({1, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({C}, flt, zero.data());
    Tensor y1 = new BufferTensor({1, H, W, C}, flt, star.data());
    D op(1, 1, SAME, 1);
    op.set_inputs({{D::in, tx}, {D::filter, tf}, {D::bias, tb}}).set_outputs({{D::out, y1}}).eval();
    affine_then_clamp(star.data(), n, C, bn.scale.data(), bn.shift.data(), 0, 6);
  });
  typedef FusedDepthwiseConv2dOperator<float> F;
  const double t_fused = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({1, K, K, C}, flt, f.data());
    Tensor tb = new RomTensor({C}, flt, bn.shift.data());
    Tensor ts = new RomTensor({C}, flt, bn.scale.data());
    Tensor ty = new BufferTensor({1, H, W, C}, flt, fused.data());
    F op(1, 1, SAME, FusedActivation::relu6());
    op.set_inputs({{F::in, tx}, {F::filter, tf}, {F::bias, tb}, {F::scale, ts}})
        .set_outputs({{F::out, ty}}).eval();
  });
  const double t_folded = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    Tensor tf = new RomTensor({1, K, K, C}, flt, ff.data());
    Tensor tb = new RomTensor({C}, flt, bn.shift.data());
    Tensor ty = new BufferTensor({1, H, W, C}, flt, folded.data());
    F op(1, 1, SAME, FusedActivation::relu6());
    op.set_inputs({{F::in, tx}, {F::filter, tf}, {F::bias, tb}}).set_outputs({{F::out, ty}}).eval();
  });
  row("unfused", base, base, "3 ops, 2 intermediate tensors (512 KB)");
  row("unfused*", t_star, base, diff_str(star, ref, false));
  row("fused", t_fused, base, diff_str(fused, ref, false));
  row("folded", t_folded, base, diff_str(folded, ref, false));
}

void dense_bn_relu() {
  const uint16_t D = 512, U = 256;
  header("fully connected 1x512 -> 256, + batch norm + ReLU");
  std::vector<float> x = rnd(D, 1, 30), w = rnd(D * U, 0.1f, 31), b = rnd(U, 0.3f, 32);
  BN bn(U);
  fusion::fold_batch_norm(bn.mean.data(), bn.var.data(), bn.gamma.data(), bn.beta.data(),
                          kEps, b.data(), U, bn.scale.data(), bn.shift.data());
  std::vector<float> wf = w;
  fusion::fold_dense_weights(wf.data(), D, U, bn.scale.data());
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> t1(U), t2(U), ref(U), star(U), fused(U), folded(U), zero(U, 0.0f);
  typedef FullyConnectedOperator<float> FC;
  const double base = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor y1 = new BufferTensor({1, U}, flt, t1.data());
    Tensor y2 = new BufferTensor({1, U}, flt, t2.data());
    Tensor y3 = new BufferTensor({1, U}, flt, ref.data());
    FC op;
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}}).set_outputs({{FC::output, y1}}).eval();
    batch_norm_op(y1, y2, bn, U);
    clip_op(y2, y3, 0, inf);
  });
  const double t_star = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, zero.data());
    Tensor y1 = new BufferTensor({1, U}, flt, star.data());
    FC op;
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}}).set_outputs({{FC::output, y1}}).eval();
    affine_then_clamp(star.data(), U, U, bn.scale.data(), bn.shift.data(), 0, inf);
  });
  typedef FusedFullyConnectedOperator<float> F;
  const double t_fused = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, bn.shift.data());
    Tensor ts = new RomTensor({U}, flt, bn.scale.data());
    Tensor ty = new BufferTensor({1, U}, flt, fused.data());
    F op(FusedActivation::relu());
    op.set_inputs({{F::input, tx}, {F::filter, tw}, {F::bias, tb}, {F::scale, ts}})
        .set_outputs({{F::output, ty}}).eval();
  });
  const double t_folded = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, wf.data());
    Tensor tb = new RomTensor({U}, flt, bn.shift.data());
    Tensor ty = new BufferTensor({1, U}, flt, folded.data());
    F op(FusedActivation::relu());
    op.set_inputs({{F::input, tx}, {F::filter, tw}, {F::bias, tb}}).set_outputs({{F::output, ty}}).eval();
  });
  row("unfused", base, base, "3 ops, 2 intermediate tensors (2 KB)");
  row("unfused*", t_star, base, diff_str(star, ref, false));
  row("fused", t_fused, base, diff_str(fused, ref, false));
  row("folded", t_folded, base, diff_str(folded, ref, false));

  // Activation only: the old operator's std::function activation vs a clamp.
  std::vector<float> a_old(U), a_new(U);
  const double t_old = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor ty = new BufferTensor({1, U}, flt, a_old.data());
    FC op(Fuseable::ReLU<float>);
    op.set_inputs({{FC::input, tx}, {FC::filter, tw}, {FC::bias, tb}}).set_outputs({{FC::output, ty}}).eval();
  });
  const double t_new = best_us([&] {
    Tensor tx = new RomTensor({1, D}, flt, x.data());
    Tensor tw = new RomTensor({D, U}, flt, w.data());
    Tensor tb = new RomTensor({U}, flt, b.data());
    Tensor ty = new BufferTensor({1, U}, flt, a_new.data());
    F op(FusedActivation::relu());
    op.set_inputs({{F::input, tx}, {F::filter, tw}, {F::bias, tb}}).set_outputs({{F::output, ty}}).eval();
  });
  std::printf("  relu only: FullyConnected(ReLU) %.1f us, fused %.1f us, %s\n", t_old, t_new,
              diff_str(a_new, a_old, true));
}

void add_relu() {
  const uint16_t H = 32, W = 32, C = 64;
  header("residual add 1x32x32x64 + ReLU");
  const uint32_t n = H * W * C;
  std::vector<float> a = rnd(n, 1, 40), b = rnd(n, 1, 41), t1(n), ref(n), star(n), fused(n);
  const float inf = std::numeric_limits<float>::infinity();
  const double base = best_us([&] {
    typedef BroadcastAddOperator<float> Op;
    Tensor ta = new RomTensor({1, H, W, C}, flt, a.data());
    Tensor tb = new RomTensor({1, H, W, C}, flt, b.data());
    Tensor y1 = new BufferTensor({1, H, W, C}, flt, t1.data());
    Tensor y2 = new BufferTensor({1, H, W, C}, flt, ref.data());
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, y1}}).eval();
    clip_op(y1, y2, 0, inf);
  });
  const double t_star = best_us([&] {
    typedef BroadcastAddOperator<float> Op;
    Tensor ta = new RomTensor({1, H, W, C}, flt, a.data());
    Tensor tb = new RomTensor({1, H, W, C}, flt, b.data());
    Tensor y1 = new BufferTensor({1, H, W, C}, flt, star.data());
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, y1}}).eval();
    for (uint32_t i = 0; i < n; ++i) star[i] = star[i] < 0 ? 0 : star[i];
  });
  const double t_fused = best_us([&] {
    typedef FusedAddOperator<float> Op;
    Tensor ta = new RomTensor({1, H, W, C}, flt, a.data());
    Tensor tb = new RomTensor({1, H, W, C}, flt, b.data());
    Tensor ty = new BufferTensor({1, H, W, C}, flt, fused.data());
    Op op(FusedActivation::relu());
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, ty}}).eval();
  });
  row("unfused", base, base, "2 ops, 1 intermediate tensor (256 KB)");
  row("unfused*", t_star, base, diff_str(star, ref, true));
  row("fused", t_fused, base, diff_str(fused, ref, true));
}

#if PTENSOR_WITH_QUANTIZED_OPS
void int8_add_relu() {
  const uint16_t H = 32, W = 32, C = 64;
  header("int8 residual add 1x32x32x64 + ReLU (TFLite semantics)");
  const uint32_t n = H * W * C;
  std::vector<int8_t> a(n), b(n), t1(n), ref(n), fused(n);
  uint32_t s = 7;
  for (uint32_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    a[i] = static_cast<int8_t>(s >> 24);
    b[i] = static_cast<int8_t>(s >> 16);
  }
  typedef TflmSymQuantOps::QuantizedAddOperator<int8_t> Q;
  auto quant = [](Tensor& ta, Tensor& tb, Tensor& tc) {
    ta->set_quantization_params(PerTensorQuantizationParams(-4, 0.05f));
    tb->set_quantization_params(PerTensorQuantizationParams(7, 0.11f));
    tc->set_quantization_params(PerTensorQuantizationParams(3, 0.157f));
  };
  const double base = best_us([&] {
    Tensor ta = new RomTensor({1, H, W, C}, i8, a.data());
    Tensor tb = new RomTensor({1, H, W, C}, i8, b.data());
    Tensor y1 = new BufferTensor({1, H, W, C}, i8, t1.data());
    Tensor y2 = new BufferTensor({1, H, W, C}, i8, ref.data());
    quant(ta, tb, y1);
    y2->set_quantization_params(PerTensorQuantizationParams(3, 0.157f));
    Q op(TFLM::kTfLiteActNone);
    op.set_inputs({{Q::a, ta}, {Q::b, tb}}).set_outputs({{Q::c, y1}}).eval();
    ReLUOperator<int8_t> relu;
    relu.set_inputs({{ReLUOperator<int8_t>::in, y1}})
        .set_outputs({{ReLUOperator<int8_t>::out, y2}}).eval();
  });
  const double t_fused = best_us([&] {
    Tensor ta = new RomTensor({1, H, W, C}, i8, a.data());
    Tensor tb = new RomTensor({1, H, W, C}, i8, b.data());
    Tensor ty = new BufferTensor({1, H, W, C}, i8, fused.data());
    quant(ta, tb, ty);
    Q op(TFLM::kTfLiteActRelu);
    op.set_inputs({{Q::a, ta}, {Q::b, tb}}).set_outputs({{Q::c, ty}}).eval();
  });
  // "add, then clamp at the zero point" is what the fused op must equal.
  uint32_t vs_clamp = 0, vs_relu_op = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const int8_t c = t1[i] < 3 ? 3 : t1[i];
    vs_clamp += fused[i] != c;
    vs_relu_op += fused[i] != ref[i];
  }
  row("unfused", base, base, "add + int8 ReLU op, 1 intermediate tensor (64 KB)");
  char msg[128];
  std::snprintf(msg, sizeof msg, "%u/%u differ from add+clamp; %u/%u from add+ReLU op",
                vs_clamp, n, vs_relu_op, n);
  row("fused", t_fused, base, msg);
}
#endif

// A MobileNetV2-style inverted-residual block, start to finish.
void mobile_block() {
  const uint16_t H = 24, W = 24, C = 16, E = 64, K = 3;
  header("block: 1x1 expand 16->64 +BN+ReLU6, 3x3 depthwise +BN+ReLU6, 1x1 project 64->16 +BN, + residual");
  std::vector<float> x = rnd(H * W * C, 1, 50);
  std::vector<float> f1 = rnd(E * C, 0.4f, 51), b1 = rnd(E, 0.2f, 52);
  std::vector<float> f2 = rnd(K * K * E, 0.5f, 53), b2 = rnd(E, 0.2f, 54);
  std::vector<float> f3 = rnd(C * E, 0.2f, 55), b3 = rnd(C, 0.2f, 56);
  BN bn1(E), bn2(E), bn3(C);
  fusion::fold_batch_norm(bn1.mean.data(), bn1.var.data(), bn1.gamma.data(), bn1.beta.data(), kEps, b1.data(), E, bn1.scale.data(), bn1.shift.data());
  fusion::fold_batch_norm(bn2.mean.data(), bn2.var.data(), bn2.gamma.data(), bn2.beta.data(), kEps, b2.data(), E, bn2.scale.data(), bn2.shift.data());
  fusion::fold_batch_norm(bn3.mean.data(), bn3.var.data(), bn3.gamma.data(), bn3.beta.data(), kEps, b3.data(), C, bn3.scale.data(), bn3.shift.data());
  std::vector<float> g1 = f1, g2 = f2, g3 = f3;
  fusion::fold_conv2d_weights(g1.data(), E, C, bn1.scale.data());
  fusion::fold_depthwise_weights(g2.data(), K * K, E, bn2.scale.data());
  fusion::fold_conv2d_weights(g3.data(), C, E, bn3.scale.data());

  const uint32_t ne = H * W * E, nc = H * W * C;
  std::vector<float> e1(ne), e2(ne), e3(ne), d1(ne), d2(ne), d3(ne), p1(nc), p2(nc), ref(nc);
  std::vector<float> fa(ne), fb(ne), fc(nc), out(nc);
  typedef Conv2dOperator<float> Conv;
  typedef DepthwiseConv2dOperator<float> DW;
  const double base = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    {
      Tensor tf = new RomTensor({E, 1, 1, C}, flt, f1.data());
      Tensor tb = new RomTensor({E}, flt, b1.data());
      Tensor y = new BufferTensor({1, H, W, E}, flt, e1.data());
      Conv op({1, 1}, SAME);
      op.set_inputs({{Conv::in, tx}, {Conv::filter, tf}, {Conv::bias, tb}}).set_outputs({{Conv::out, y}}).eval();
      Tensor y2 = new BufferTensor({1, H, W, E}, flt, e2.data());
      Tensor y3 = new BufferTensor({1, H, W, E}, flt, e3.data());
      batch_norm_op(y, y2, bn1, E);
      clip_op(y2, y3, 0, 6);
    }
    {
      Tensor ti = new RomTensor({1, H, W, E}, flt, e3.data());
      Tensor tf = new RomTensor({1, K, K, E}, flt, f2.data());
      Tensor tb = new RomTensor({E}, flt, b2.data());
      Tensor y = new BufferTensor({1, H, W, E}, flt, d1.data());
      DW op(1, 1, SAME, 1);
      op.set_inputs({{DW::in, ti}, {DW::filter, tf}, {DW::bias, tb}}).set_outputs({{DW::out, y}}).eval();
      Tensor y2 = new BufferTensor({1, H, W, E}, flt, d2.data());
      Tensor y3 = new BufferTensor({1, H, W, E}, flt, d3.data());
      batch_norm_op(y, y2, bn2, E);
      clip_op(y2, y3, 0, 6);
    }
    {
      Tensor ti = new RomTensor({1, H, W, E}, flt, d3.data());
      Tensor tf = new RomTensor({C, 1, 1, E}, flt, f3.data());
      Tensor tb = new RomTensor({C}, flt, b3.data());
      Tensor y = new BufferTensor({1, H, W, C}, flt, p1.data());
      Conv op({1, 1}, SAME);
      op.set_inputs({{Conv::in, ti}, {Conv::filter, tf}, {Conv::bias, tb}}).set_outputs({{Conv::out, y}}).eval();
      Tensor y2 = new BufferTensor({1, H, W, C}, flt, p2.data());
      batch_norm_op(y, y2, bn3, C);
      typedef BroadcastAddOperator<float> Add;
      Tensor yr = new BufferTensor({1, H, W, C}, flt, ref.data());
      Add add;
      add.set_inputs({{Add::a, y2}, {Add::b, tx}}).set_outputs({{Add::c, yr}}).eval();
    }
  });
  typedef FusedConv2dOperator<float> FConv;
  typedef FusedDepthwiseConv2dOperator<float> FDW;
  // The 1x1 projection has no activation, so the residual add is a separate
  // (plain) add: fusing it into the conv would need a second input.
  const double t_fused = best_us([&] {
    Tensor tx = new RomTensor({1, H, W, C}, flt, x.data());
    {
      Tensor tf = new RomTensor({E, 1, 1, C}, flt, g1.data());
      Tensor tb = new RomTensor({E}, flt, bn1.shift.data());
      Tensor y = new BufferTensor({1, H, W, E}, flt, fa.data());
      FConv op(1, 1, SAME, FusedActivation::relu6());
      op.set_inputs({{FConv::in, tx}, {FConv::filter, tf}, {FConv::bias, tb}}).set_outputs({{FConv::out, y}}).eval();
    }
    {
      Tensor ti = new RomTensor({1, H, W, E}, flt, fa.data());
      Tensor tf = new RomTensor({1, K, K, E}, flt, g2.data());
      Tensor tb = new RomTensor({E}, flt, bn2.shift.data());
      Tensor y = new BufferTensor({1, H, W, E}, flt, fb.data());
      FDW op(1, 1, SAME, FusedActivation::relu6());
      op.set_inputs({{FDW::in, ti}, {FDW::filter, tf}, {FDW::bias, tb}}).set_outputs({{FDW::out, y}}).eval();
    }
    {
      Tensor ti = new RomTensor({1, H, W, E}, flt, fb.data());
      Tensor tf = new RomTensor({C, 1, 1, E}, flt, g3.data());
      Tensor tb = new RomTensor({C}, flt, bn3.shift.data());
      Tensor y = new BufferTensor({1, H, W, C}, flt, fc.data());
      FConv op(1, 1, SAME);
      op.set_inputs({{FConv::in, ti}, {FConv::filter, tf}, {FConv::bias, tb}}).set_outputs({{FConv::out, y}}).eval();
      typedef FusedAddOperator<float> Add;
      Tensor yr = new BufferTensor({1, H, W, C}, flt, out.data());
      Add add;
      add.set_inputs({{Add::a, y}, {Add::b, tx}}).set_outputs({{Add::c, yr}}).eval();
    }
  });
  row("unfused", base, base, "9 ops, 8 intermediate tensors (0.96 MB)");
  row("folded", t_fused, base, diff_str(out, ref, false));
  std::printf("  fused: 4 ops, 3 intermediate tensors (0.33 MB)\n");
}

}  // namespace

int main() {
  Context::get_default_context()->set_metadata_allocator(&meta);
  Context::get_default_context()->set_ram_data_allocator(&ram);
  Context::get_default_context()->set_ErrorHandler(&err);
  std::printf("best of N runs; speed-up is relative to 'unfused'\n");
  conv_bn_relu6();
  depthwise_bn_relu6();
  dense_bn_relu();
  add_relu();
#if PTENSOR_WITH_QUANTIZED_OPS
  int8_add_relu();
#endif
  mobile_block();
  return 0;
}
