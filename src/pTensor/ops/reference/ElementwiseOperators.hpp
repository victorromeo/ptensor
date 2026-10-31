#ifndef PTENSOR_ELEMENTWISE_OPS_H
#define PTENSOR_ELEMENTWISE_OPS_H
/**
 * Elementwise unary and binary operators.
 *
 * Each operator is a header-only template: one you never instantiate adds
 * nothing to the binary. The math lives in small functors so the compiler can
 * inline it into the loop (no std::function or virtual call per element).
 *
 * Binary operators follow NumPy broadcasting over up to 4 dimensions: shapes
 * are aligned from the right and a dimension of size 1 is repeated.
 */
#include <cmath>

#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {

DECLARE_ERROR(BroadcastShapeMismatchError);

namespace elementwise {

// Right-aligned 4D view of a shape with per-dimension strides (0 where the
// dimension is broadcast).
struct BroadcastView {
  uint32_t dims[4];
  uint32_t strides[4];
};

inline void right_align(const TensorShape& s, uint32_t (&dims)[4]) {
  const uint8_t n = s.num_dims();
  for (int i = 0; i < 4; ++i) dims[i] = 1;
  for (int i = 0; i < n; ++i) dims[4 - n + i] = s[i] ? s[i] : 1;
}

inline BroadcastView make_view(const TensorShape& in, const uint32_t (&out)[4],
                               bool& ok) {
  BroadcastView v;
  right_align(in, v.dims);
  uint32_t stride = 1;
  for (int i = 3; i >= 0; --i) {
    if (v.dims[i] == out[i]) {
      v.strides[i] = stride;
    } else if (v.dims[i] == 1) {
      v.strides[i] = 0;
    } else {
      ok = false;
      v.strides[i] = 0;
    }
    stride *= v.dims[i];
  }
  return v;
}

template <typename T>
struct Add {
  T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct Sub {
  T operator()(T a, T b) const { return a - b; }
};
template <typename T>
struct Mul {
  T operator()(T a, T b) const { return a * b; }
};
template <typename T>
struct Div {
  T operator()(T a, T b) const { return a / b; }
};
template <typename T>
struct Maximum {
  T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct Minimum {
  T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct SquaredDifference {
  T operator()(T a, T b) const { return (a - b) * (a - b); }
};
template <typename T>
struct Pow {
  T operator()(T a, T b) const { return static_cast<T>(std::pow(a, b)); }
};

}  // namespace elementwise

namespace ReferenceOperators {

/**
 * out = Fn(a, b) with broadcasting. The output tensor must already have the
 * broadcast shape.
 */
template <typename T, typename Fn>
class BinaryElementwiseOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };

 protected:
  virtual void compute() {
    const Tensor& ta = inputs[a].tensor();
    const Tensor& tb = inputs[b].tensor();
    Tensor& tc = outputs[c].tensor();
    uint32_t od[4];
    elementwise::right_align(tc->get_shape(), od);
    bool ok = true;
    const elementwise::BroadcastView va =
        elementwise::make_view(ta->get_shape(), od, ok);
    const elementwise::BroadcastView vb =
        elementwise::make_view(tb->get_shape(), od, ok);
    if (!ok) {
      Context::get_default_context()->throwError(BroadcastShapeMismatchError());
      return;
    }
    Fn fn;
    uint32_t oi = 0;
    for (uint32_t i0 = 0; i0 < od[0]; ++i0)
      for (uint32_t i1 = 0; i1 < od[1]; ++i1)
        for (uint32_t i2 = 0; i2 < od[2]; ++i2) {
          const uint32_t base_a =
              i0 * va.strides[0] + i1 * va.strides[1] + i2 * va.strides[2];
          const uint32_t base_b =
              i0 * vb.strides[0] + i1 * vb.strides[1] + i2 * vb.strides[2];
          for (uint32_t i3 = 0; i3 < od[3]; ++i3, ++oi) {
            const T x = ta(base_a + i3 * va.strides[3]);
            const T y = tb(base_b + i3 * vb.strides[3]);
            tc(oi) = fn(x, y);
          }
        }
  }
};

template <typename T>
using SubOperator = BinaryElementwiseOperator<T, elementwise::Sub<T> >;
template <typename T>
using MulOperator = BinaryElementwiseOperator<T, elementwise::Mul<T> >;
template <typename T>
using DivOperator = BinaryElementwiseOperator<T, elementwise::Div<T> >;
template <typename T>
using MaximumOperator = BinaryElementwiseOperator<T, elementwise::Maximum<T> >;
template <typename T>
using MinimumOperator = BinaryElementwiseOperator<T, elementwise::Minimum<T> >;
template <typename T>
using SquaredDifferenceOperator =
    BinaryElementwiseOperator<T, elementwise::SquaredDifference<T> >;
template <typename T>
using PowOperator = BinaryElementwiseOperator<T, elementwise::Pow<T> >;
/** Add with broadcasting (the older AddOperator needs equal shapes). */
template <typename T>
using BroadcastAddOperator = BinaryElementwiseOperator<T, elementwise::Add<T> >;

/** out = Fn(in) elementwise. Fn may carry parameters (see LeakyReLU). */
template <typename T, typename Fn>
class UnaryElementwiseOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  UnaryElementwiseOperator() : _fn() {}
  explicit UnaryElementwiseOperator(const Fn& fn) : _fn(fn) {}

 protected:
  virtual void compute() {
    const Tensor& ti = inputs[in].tensor();
    Tensor& to = outputs[out].tensor();
    const uint32_t n = ti->get_shape().get_linear_size();
    if (n != to->get_shape().get_linear_size()) {
      Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
      return;
    }
    for (uint32_t i = 0; i < n; ++i) {
      const T x = ti(i);
      to(i) = _fn(x);
    }
  }

 private:
  Fn _fn;
};

}  // namespace ReferenceOperators

namespace activations {

struct LeakyReLU {
  float alpha;
  LeakyReLU(float alpha = 0.2f) : alpha(alpha) {}
  float operator()(float x) const { return x >= 0.0f ? x : alpha * x; }
};
struct ELU {
  float alpha;
  ELU(float alpha = 1.0f) : alpha(alpha) {}
  float operator()(float x) const {
    return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
  }
};
struct SELU {
  float operator()(float x) const {
    const float a = 1.6732632423543772f, s = 1.0507009873554805f;
    return s * (x >= 0.0f ? x : a * (std::exp(x) - 1.0f));
  }
};
/** TFLite / Keras hard sigmoid: clamp(x / 6 + 0.5, 0, 1) (MobileNetV3 form). */
struct HardSigmoid {
  float operator()(float x) const {
    const float y = x * (1.0f / 6.0f) + 0.5f;
    return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
  }
};
struct HardSwish {
  float operator()(float x) const { return x * HardSigmoid()(x); }
};
/** Swish / SiLU: x * sigmoid(x). */
struct Swish {
  float operator()(float x) const { return x / (1.0f + std::exp(-x)); }
};
/** GELU, tanh approximation (as used by most exported models). */
struct GELU {
  float operator()(float x) const {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
  }
};
struct Softplus {
  float operator()(float x) const {
    // log(1 + e^x), stable for large |x|
    return x > 20.0f ? x : (x < -20.0f ? std::exp(x) : std::log1p(std::exp(x)));
  }
};
struct Softsign {
  float operator()(float x) const { return x / (1.0f + std::fabs(x)); }
};
struct Abs {
  float operator()(float x) const { return std::fabs(x); }
};
struct Neg {
  float operator()(float x) const { return -x; }
};
struct Square {
  float operator()(float x) const { return x * x; }
};
struct Sqrt {
  float operator()(float x) const { return std::sqrt(x); }
};
struct Rsqrt {
  float operator()(float x) const { return 1.0f / std::sqrt(x); }
};
struct Exp {
  float operator()(float x) const { return std::exp(x); }
};
struct Log {
  float operator()(float x) const { return std::log(x); }
};
struct Floor {
  float operator()(float x) const { return std::floor(x); }
};
struct Ceil {
  float operator()(float x) const { return std::ceil(x); }
};
struct Round {
  // Round half to even, matching TFLite/NumPy.
  float operator()(float x) const { return std::nearbyint(x); }
};
struct Sin {
  float operator()(float x) const { return std::sin(x); }
};
struct Cos {
  float operator()(float x) const { return std::cos(x); }
};
/** Clamp to [lo, hi] (ReLU-N1-to-1 is Clip(-1, 1)). */
struct Clip {
  float lo, hi;
  Clip(float lo = -1.0f, float hi = 1.0f) : lo(lo), hi(hi) {}
  float operator()(float x) const { return x < lo ? lo : (x > hi ? hi : x); }
};

}  // namespace activations

namespace ReferenceOperators {

#define PTENSOR_UNARY_OP(NAME, FN) \
  using NAME = UnaryElementwiseOperator<float, activations::FN>

PTENSOR_UNARY_OP(LeakyReLUOperator, LeakyReLU);
PTENSOR_UNARY_OP(EluOperator, ELU);
PTENSOR_UNARY_OP(SeluOperator, SELU);
PTENSOR_UNARY_OP(HardSigmoidOperator, HardSigmoid);
PTENSOR_UNARY_OP(HardSwishOperator, HardSwish);
PTENSOR_UNARY_OP(SwishOperator, Swish);
PTENSOR_UNARY_OP(GeluOperator, GELU);
PTENSOR_UNARY_OP(SoftplusOperator, Softplus);
PTENSOR_UNARY_OP(SoftsignOperator, Softsign);
PTENSOR_UNARY_OP(AbsOperator, Abs);
PTENSOR_UNARY_OP(NegOperator, Neg);
PTENSOR_UNARY_OP(SquareOperator, Square);
PTENSOR_UNARY_OP(SqrtOperator, Sqrt);
PTENSOR_UNARY_OP(RsqrtOperator, Rsqrt);
PTENSOR_UNARY_OP(ExpOperator, Exp);
PTENSOR_UNARY_OP(LogOperator, Log);
PTENSOR_UNARY_OP(FloorOperator, Floor);
PTENSOR_UNARY_OP(CeilOperator, Ceil);
PTENSOR_UNARY_OP(RoundOperator, Round);
PTENSOR_UNARY_OP(SinOperator, Sin);
PTENSOR_UNARY_OP(CosOperator, Cos);
PTENSOR_UNARY_OP(ClipOperator, Clip);

#undef PTENSOR_UNARY_OP

}  // namespace ReferenceOperators
}  // namespace pTensor
#endif
