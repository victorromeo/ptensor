#ifndef PTENSOR_UTIL_FIXED_POINT_H
#define PTENSOR_UTIL_FIXED_POINT_H
/**
 * Header-only TFLite / gemmlowp fixed-point helpers, for kernels that must
 * reproduce TFLite's integer arithmetic bit for bit without linking the
 * quantized-ops library (ops/tflm/*.cpp, PTENSOR_WITH_QUANTIZED_OPS).
 *
 * Same results as gemmlowp::SaturatingRoundingDoublingHighMul,
 * gemmlowp::RoundingDivideByPOT, TFLM::MultiplyByQuantizedMultiplier and
 * TFLM::QuantizeMultiplier, but inline and without error reporting: callers
 * check quantize_multiplier()'s result once, before the element loop.
 */
#include <cmath>
#include <limits>
#include <stdint.h>

namespace pTensor {
namespace fixed_point {

inline int32_t saturating_rounding_doubling_high_mul(int32_t a, int32_t b) {
  const bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  const int64_t ab = static_cast<int64_t>(a) * static_cast<int64_t>(b);
  const int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  const int32_t hi = static_cast<int32_t>((ab + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : hi;
}

/** x / 2^exponent, rounded half away from zero; exponent in [0, 31]. */
inline int32_t rounding_divide_by_pot(int32_t x, int32_t exponent) {
  const int32_t mask = static_cast<int32_t>((1ll << exponent) - 1);
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

/** x * multiplier * 2^shift in Q31 (TFLite's MultiplyByQuantizedMultiplier). */
inline int32_t multiply_by_quantized_multiplier(int32_t x, int32_t multiplier,
                                                int32_t shift) {
  const int32_t left = shift > 0 ? shift : 0;
  const int32_t right = shift > 0 ? 0 : -shift;
  return rounding_divide_by_pot(
      saturating_rounding_doubling_high_mul(x * (1 << left), multiplier), right);
}

/** shift is usable by multiply_by_quantized_multiplier(). */
inline bool shift_ok(int32_t shift) { return shift >= -31 && shift <= 30; }

/**
 * real = multiplier * 2^shift / 2^31 (TFLite's QuantizeMultiplier). Returns
 * false where TFLite would raise an error (multiplier out of range).
 */
inline bool quantize_multiplier(double real, int32_t* multiplier, int32_t* shift) {
  if (real == 0.0) {
    *multiplier = 0;
    *shift = 0;
    return true;
  }
  int s = 0;
  const double q = std::frexp(real, &s);
  int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
  if (q_fixed > (1ll << 31)) return false;
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++s;
  }
  if (q_fixed > std::numeric_limits<int32_t>::max()) return false;
  if (s < -31) {
    s = 0;
    q_fixed = 0;
  }
  *multiplier = static_cast<int32_t>(q_fixed);
  *shift = s;
  return shift_ok(s);
}

}  // namespace fixed_point
}  // namespace pTensor
#endif
