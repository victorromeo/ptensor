#ifndef PTENSOR_UTIL_POLYQUANT_HPP
#define PTENSOR_UTIL_POLYQUANT_HPP
/**
 * Polynomial (quadratic / cubic) quantisation: the decoder.
 *
 * Linear quantisation maps an integer code k to a real value with a straight
 * line, real = (k - zero_point) * scale. Polynomial quantisation replaces the
 * line with a polynomial of degree 1..3 in the normalised code
 *
 *     t = k * (2 / (L - 1)) - 1          (L = 2^bits, so t runs from -1 to 1)
 *     real = ((c3 * t + c2) * t + c1) * t + c0
 *
 * so the levels can bunch up where the weights are dense (near zero) and
 * spread out in the tails. The coefficients are fitted per tensor by the
 * encoder (python/ptensor/polyquant.py) during quantisation-aware training.
 *
 * On the device the polynomial is normally materialised once as a lookup
 * table of L floats (in flash, or built into RAM with build_lut()), so
 * decoding a weight costs one bit-stream read and one table lookup, the same
 * as linear PackedRomTensor decoding. Without a table (sensible only for 16-bit
 * codes, whose table would be 256 KB) the polynomial is evaluated directly.
 *
 * All arithmetic is float32 in exactly this order, which the Python encoder
 * reproduces, so tables built here and in Python are identical.
 *
 * Header-only and allocation-free.
 */
#include <cstddef>
#include <cstdint>

#include "pTensor/util/packing.hpp"

namespace pTensor {
namespace polyquant {

/** Normalised position of code `k` for `bits`-bit codes, in [-1, 1]. */
inline float code_to_t(uint32_t k, uint8_t bits) {
  const float levels_minus_one = static_cast<float>((1u << bits) - 1u);
  const float step = 2.0f / levels_minus_one;
  const float scaled = static_cast<float>(k) * step;
  return scaled - 1.0f;
}

/** The polynomial at t (Horner's rule, float32). coeffs = c0, c1, c2, c3. */
inline float eval(const float* coeffs, uint8_t degree, float t) {
  float r = coeffs[degree];
  for (int i = static_cast<int>(degree) - 1; i >= 0; --i) {
    const float m = r * t;
    r = m + coeffs[i];
  }
  return r;
}

/** Value of code `k`. */
inline float decode_code(const float* coeffs, uint8_t degree, uint8_t bits,
                         uint32_t k) {
  return eval(coeffs, degree, code_to_t(k, bits));
}

/** Fill `lut` (2^bits floats) with the value of every code. */
inline void build_lut(const float* coeffs, uint8_t degree, uint8_t bits,
                      float* lut) {
  const uint32_t n = 1u << bits;
  for (uint32_t k = 0; k < n; ++k) lut[k] = decode_code(coeffs, degree, bits, k);
}

/** Bytes of code stream for `count` codes. */
inline size_t stream_bytes(size_t count, uint8_t bits) {
  return packing::packed_size_bytes(count, bits);
}

}  // namespace polyquant
}  // namespace pTensor
#endif
