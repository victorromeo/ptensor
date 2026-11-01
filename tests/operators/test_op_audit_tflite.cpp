// Operator audit, int8: pTensor against TFLite's reference kernels, one
// single-operator TFLite model per case (constants from
// gen_constants_op_audit_tflite.py). Every case must be bit-exact, including
// LOGISTIC / TANH (checked on all 256 int8 inputs) and SOFTMAX, which pTensor
// computes in float and TFLite with fixed-point lookups.
#include <cstdlib>
#include <vector>

#include "constants_op_audit_tflite.hpp"
#include "gtest/gtest.h"
#include "pTensor.h"

using namespace pTensor;

namespace {

struct TfliteAudit : public ::testing::Test {
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

template <size_t N>
void expect_close(const std::vector<int8_t>& got, const int8_t (&ref)[N], int tol,
                  const char* name) {
  ASSERT_EQ(got.size(), N) << name;
  int worst = 0;
  size_t mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    const int d = std::abs(static_cast<int>(got[i]) - ref[i]);
    worst = d > worst ? d : worst;
    if (d) ++mismatches;
  }
  EXPECT_LE(worst, tol) << name << ": " << mismatches << " of " << N << " differ";
}

}  // namespace

#if PTENSOR_WITH_QUANTIZED_OPS

namespace {

// TflmSymQuantOps::Conv2dOperator with TFLite's OHWI filter.
template <size_t NI, size_t NO, size_t NW, size_t NS, size_t NB>
std::vector<int8_t> run_conv(const int8_t (&x)[NI], float xs, int32_t xz,
                             const int8_t (&w)[NW], const float (&ws)[NS],
                             const int32_t (&b)[NB], float ys, int32_t yz,
                             TensorShape is, TensorShape fs, TensorShape os,
                             uint16_t sh, uint16_t sw, Padding p, uint16_t dil, const int8_t (&)[NO]) {
  std::vector<int8_t> y(NO, 0);
  std::vector<int32_t> zeros(NS, 0);
  std::vector<float> bs(NS);
  for (size_t i = 0; i < NS; ++i) bs[i] = xs * ws[i];
  {
    Tensor tx = new RomTensor(is, i8, x);
    tx->set_quantization_params(PerTensorQuantizationParams(xz, xs));
    Tensor tw = new RomTensor(fs, i8, w);
    tw->set_quantization_params(PerChannelQuantizationParams(zeros.data(), ws, NS));
    Tensor tb = new RomTensor({(uint16_t)NB}, i32, b);
    tb->set_quantization_params(PerChannelQuantizationParams(zeros.data(), bs.data(), NS));
    Tensor ty = new BufferTensor(os, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(yz, ys));
    const uint16_t strides[2] = {sh, sw};
    const uint16_t dilation[2] = {dil, dil};
    typedef TflmSymQuantOps::Conv2dOperator<int8_t> Op;
    Op op(strides, p, dilation);
    op.set_inputs({{Op::in, tx}, {Op::filter, tw}, {Op::bias, tb}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  return y;
}

template <size_t NI, size_t NO, size_t NW, size_t NS, size_t NB>
std::vector<int8_t> run_dw(const int8_t (&x)[NI], float xs, int32_t xz,
                           const int8_t (&w)[NW], const float (&ws)[NS],
                           const int32_t (&b)[NB], float ys, int32_t yz,
                           TensorShape is, TensorShape fs, TensorShape os,
                           uint16_t sh, uint16_t sw, Padding p, int mult, uint16_t dil,
                           const int8_t (&)[NO], bool per_tensor = false) {
  std::vector<int8_t> y(NO, 0);
  std::vector<int32_t> zeros(NS, 0);
  std::vector<float> bs(NS);
  for (size_t i = 0; i < NS; ++i) bs[i] = xs * ws[i];
  {
    Tensor tx = new RomTensor(is, i8, x);
    tx->set_quantization_params(PerTensorQuantizationParams(xz, xs));
    Tensor tw = new RomTensor(fs, i8, w);
    if (per_tensor)
      tw->set_quantization_params(PerTensorQuantizationParams(0, ws[0]));
    else
      tw->set_quantization_params(PerChannelQuantizationParams(zeros.data(), ws, NS));
    Tensor tb = new RomTensor({(uint16_t)NB}, i32, b);
    tb->set_quantization_params(PerChannelQuantizationParams(zeros.data(), bs.data(), NS));
    Tensor ty = new BufferTensor(os, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(yz, ys));
    const uint16_t strides[2] = {sh, sw};
    const uint16_t dilation[2] = {dil, dil};
    typedef TflmSymQuantOps::DepthwiseSeparableConvOperator<int8_t> Op;
    Op op(strides, p, mult, dilation);
    op.set_inputs({{Op::in, tx}, {Op::filter, tw}, {Op::bias, tb}})
        .set_outputs({{Op::out, ty}})
        .eval();
  }
  return y;
}

}  // namespace

#define CONV_CASE(NAME, IS, FS, OS, SH, SW, PAD, DIL)                               \
  TEST_F(TfliteAudit, NAME) {                                                       \
    const auto y = run_conv(NAME##_in, NAME##_in_scale, NAME##_in_zp, NAME##_w,     \
                            NAME##_w_scale, NAME##_b, NAME##_out_scale,             \
                            NAME##_out_zp, TensorShape IS, TensorShape FS,          \
                            TensorShape OS, SH, SW, PAD, DIL, NAME##_out);          \
    expect_close(y, NAME##_out, 0, #NAME);                                          \
    EXPECT_EQ(errors, 0);                                                           \
  }

CONV_CASE(conv_same, (1, 7, 6, 3), (4, 3, 3, 3), (1, 7, 6, 4), 1, 1, SAME, 1)
CONV_CASE(conv_dil2, (1, 8, 7, 2), (4, 3, 3, 2), (1, 8, 7, 4), 1, 1, SAME, 2)
CONV_CASE(conv_1x1_s4, (1, 8, 8, 2), (3, 1, 1, 2), (1, 2, 2, 3), 4, 4, SAME, 1)
CONV_CASE(conv_2x2_s4_valid, (1, 11, 11, 2), (3, 2, 2, 2), (1, 3, 3, 3), 4, 4, VALID, 1)
CONV_CASE(conv_2x2_s4_same, (1, 8, 8, 2), (3, 2, 2, 2), (1, 2, 2, 3), 4, 4, SAME, 1)
CONV_CASE(conv_s21, (1, 7, 6, 2), (3, 3, 3, 2), (1, 4, 6, 3), 2, 1, SAME, 1)

TEST_F(TfliteAudit, dw_dil2) {
  const auto y = run_dw(dw_dil2_in, dw_dil2_in_scale, dw_dil2_in_zp, dw_dil2_w, dw_dil2_w_scale,
                        dw_dil2_b, dw_dil2_out_scale, dw_dil2_out_zp, TensorShape(1, 8, 7, 3),
                        TensorShape(1, 3, 3, 3), TensorShape(1, 8, 7, 3), 1, 1, SAME, 1, 2, dw_dil2_out);
  expect_close(y, dw_dil2_out, 0, "dw_dil2");
  EXPECT_EQ(errors, 0);
}

TEST_F(TfliteAudit, dw_s2_m2) {
  const auto y = run_dw(dw_s2_m2_in, dw_s2_m2_in_scale, dw_s2_m2_in_zp, dw_s2_m2_w,
                        dw_s2_m2_w_scale, dw_s2_m2_b, dw_s2_m2_out_scale, dw_s2_m2_out_zp,
                        TensorShape(1, 7, 7, 2), TensorShape(1, 3, 3, 4), TensorShape(1, 4, 4, 4),
                        2, 2, SAME, 2, 1, dw_s2_m2_out);
  expect_close(y, dw_s2_m2_out, 0, "dw_s2_m2");
  EXPECT_EQ(errors, 0);
}

TEST_F(TfliteAudit, dw_s12) {
  const auto y = run_dw(dw_s12_in, dw_s12_in_scale, dw_s12_in_zp, dw_s12_w, dw_s12_w_scale,
                        dw_s12_b, dw_s12_out_scale, dw_s12_out_zp, TensorShape(1, 6, 9, 2),
                        TensorShape(1, 3, 3, 2), TensorShape(1, 6, 5, 2), 1, 2, SAME, 1, 1, dw_s12_out);
  expect_close(y, dw_s12_out, 0, "dw_s12");
  EXPECT_EQ(errors, 0);
}

// Per-tensor weights are broadcast like TFLite: same result as per-channel
// weights whose scales are all equal.
TEST_F(TfliteAudit, dw_per_tensor_weights) {
  float same_scale[4];
  for (int i = 0; i < 4; ++i) same_scale[i] = dw_s2_m2_w_scale[0];
  const auto per_channel =
      run_dw(dw_s2_m2_in, dw_s2_m2_in_scale, dw_s2_m2_in_zp, dw_s2_m2_w, same_scale, dw_s2_m2_b,
             dw_s2_m2_out_scale, dw_s2_m2_out_zp, TensorShape(1, 7, 7, 2),
             TensorShape(1, 3, 3, 4), TensorShape(1, 4, 4, 4), 2, 2, SAME, 2, 1, dw_s2_m2_out);
  const auto per_tensor =
      run_dw(dw_s2_m2_in, dw_s2_m2_in_scale, dw_s2_m2_in_zp, dw_s2_m2_w, same_scale, dw_s2_m2_b,
             dw_s2_m2_out_scale, dw_s2_m2_out_zp, TensorShape(1, 7, 7, 2),
             TensorShape(1, 3, 3, 4), TensorShape(1, 4, 4, 4), 2, 2, SAME, 2, 1, dw_s2_m2_out,
             true);
  EXPECT_EQ(per_tensor, per_channel);
  EXPECT_EQ(errors, 0);
}

#endif  // PTENSOR_WITH_QUANTIZED_OPS

namespace {

template <typename Op, size_t NI, size_t NO>
std::vector<int8_t> run_pool(const int8_t (&x)[NI], TensorShape is, TensorShape os,
                             uint16_t k, uint16_t s, Padding p, float scale, int32_t zp) {
  std::vector<int8_t> y(NO, 0);
  {
    Tensor tx = new RomTensor(is, i8, x);
    tx->set_quantization_params(PerTensorQuantizationParams(zp, scale));
    Tensor ty = new BufferTensor(os, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(zp, scale));
    Op op({k, k}, {1, s, s, 1}, p);
    op.set_inputs({{Op::in, tx}}).set_outputs({{Op::out, ty}}).eval();
  }
  return y;
}

}  // namespace

#define POOL_CASE(NAME, OP, IS, OS, K, S, PAD)                                       \
  TEST_F(TfliteAudit, NAME) {                                                        \
    ASSERT_EQ(NAME##_in_zp, NAME##_out_zp);                                          \
    const auto y = run_pool<ReferenceOperators::OP<int8_t>, sizeof(NAME##_in),      \
                            sizeof(NAME##_out)>(NAME##_in, TensorShape IS,          \
                                                TensorShape OS, K, S, PAD,          \
                                                NAME##_in_scale, NAME##_in_zp);     \
    expect_close(y, NAME##_out, 0, #NAME);                                           \
    EXPECT_EQ(errors, 0);                                                            \
  }

POOL_CASE(maxpool_same, MaxPoolOperator, (1, 7, 7, 2), (1, 4, 4, 2), 3, 2, SAME)
POOL_CASE(maxpool_valid, MaxPoolOperator, (1, 6, 5, 3), (1, 3, 2, 3), 2, 2, VALID)
POOL_CASE(avgpool_same, AvgPoolOperator, (1, 7, 7, 2), (1, 4, 4, 2), 3, 2, SAME)
POOL_CASE(avgpool_valid, AvgPoolOperator, (1, 6, 5, 3), (1, 3, 2, 3), 2, 2, VALID)

TEST_F(TfliteAudit, logistic) {
  std::vector<int8_t> y(256, 0);
  {
    Tensor tx = new RomTensor({256}, i8, logistic_in);
    tx->set_quantization_params(PerTensorQuantizationParams(logistic_in_zp, logistic_in_scale));
    Tensor ty = new BufferTensor({256}, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(logistic_out_zp, logistic_out_scale));
    ReferenceOperators::LogisticOperator<int8_t> op;
    op.set_inputs({{ReferenceOperators::LogisticOperator<int8_t>::in, tx}})
        .set_outputs({{ReferenceOperators::LogisticOperator<int8_t>::out, ty}})
        .eval();
  }
  expect_close(y, logistic_out, 0, "logistic");
  EXPECT_EQ(errors, 0);
}

TEST_F(TfliteAudit, tanh) {
  std::vector<int8_t> y(256, 0);
  {
    Tensor tx = new RomTensor({256}, i8, tanh_in);
    tx->set_quantization_params(PerTensorQuantizationParams(tanh_in_zp, tanh_in_scale));
    Tensor ty = new BufferTensor({256}, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(tanh_out_zp, tanh_out_scale));
    typedef ReferenceOperators::TanhOperator<int8_t> Op;
    Op op;
    op.set_inputs({{Op::act_in, tx}}).set_outputs({{Op::act_out, ty}}).eval();
  }
  expect_close(y, tanh_out, 0, "tanh");
  EXPECT_EQ(errors, 0);
}

namespace {
template <size_t N>
std::vector<int8_t> run_softmax(const int8_t (&x)[N], TensorShape s, float xs, int32_t xz,
                                float ys, int32_t yz) {
  std::vector<int8_t> y(N, 0);
  {
    Tensor tx = new RomTensor(s, i8, x);
    tx->set_quantization_params(PerTensorQuantizationParams(xz, xs));
    Tensor ty = new BufferTensor(s, i8, y.data());
    ty->set_quantization_params(PerTensorQuantizationParams(yz, ys));
    typedef ReferenceOperators::SoftmaxOperator<int8_t> Op;
    Op op;
    op.set_inputs({{Op::in, tx}}).set_outputs({{Op::out, ty}}).eval();
  }
  return y;
}
}  // namespace

TEST_F(TfliteAudit, softmax_2d) {
  const auto y = run_softmax(softmax_2d_in, TensorShape(1, 10), softmax_2d_in_scale,
                             softmax_2d_in_zp, softmax_2d_out_scale, softmax_2d_out_zp);
  expect_close(y, softmax_2d_out, 0, "softmax_2d");
  EXPECT_EQ(errors, 0);
}

TEST_F(TfliteAudit, softmax_4d) {
  const auto y = run_softmax(softmax_4d_in, TensorShape(1, 2, 3, 6), softmax_4d_in_scale,
                             softmax_4d_in_zp, softmax_4d_out_scale, softmax_4d_out_zp);
  expect_close(y, softmax_4d_out, 0, "softmax_4d");
  EXPECT_EQ(errors, 0);
}
