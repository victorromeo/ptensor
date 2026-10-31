#ifndef PTENSOR_UTIL_HOUGH_HPP
#define PTENSOR_UTIL_HOUGH_HPP
/**
 * Hough line-codebook weight storage (device-side decoding).
 *
 * The encoder (python/ptensor/hough.py) groups a tensor's values into pairs,
 * treats every pair as a point (x, y) in the plane, and runs a Hough
 * transform over those points to find the K straight lines that pass closest
 * to most of them. The lines are refined by k-lines clustering and become a
 * codebook. Each pair is then stored as
 *
 *   line index  (line_bits bits, K = 2^line_bits lines)
 *   position t  (t_bits bits, a uniform step along that line's segment)
 *
 * so a pair costs line_bits + t_bits bits, i.e. (line_bits + t_bits) / 2 bits
 * per value, plus 16 bytes per line for the codebook.
 *
 * A line with Hough parameters (theta, rho) and a quantized segment
 * [t0, t0 + (2^t_bits - 1) * step] is stored in decoder form as four floats
 *
 *   ax = rho*cos(theta) - t0*sin(theta)     dx = -step*sin(theta)
 *   ay = rho*sin(theta) + t0*cos(theta)     dy =  step*cos(theta)
 *
 * so decoding code (k, tc) is two multiply-adds: x = ax + tc*dx,
 * y = ay + tc*dy.
 *
 * Pairing. Values i and i + pair_stride form a pair, in blocks of
 * 2*pair_stride values; values after the last whole block pair with their
 * neighbour (stride 1), and an odd final value pairs with an unused slot.
 * Choosing pair_stride lets the encoder pair values that are correlated (for
 * example neighbouring filter taps), which is where the Hough lines find
 * structure.
 *
 * Bit stream: pair p's code is (k | tc << line_bits), stored as value p of an
 * LSB-first bit stream of (line_bits + t_bits)-bit values (packing.hpp).
 */
#include <cstddef>
#include <cstdint>

#include "pTensor/util/packing.hpp"

namespace pTensor {
namespace hough {

/** Floats per codebook line: ax, ay, dx, dy. */
static const uint32_t kLineFloats = 4;

/** Number of stored pairs for `count` values. */
inline uint32_t pair_count(uint32_t count, uint32_t stride) {
  const uint32_t block = 2u * stride;
  const uint32_t full = count / block;
  const uint32_t tail = count - full * block;
  return full * stride + (tail + 1u) / 2u;
}

/** Bytes of the code bit stream (the codebook is stored separately). */
inline size_t stream_bytes(uint32_t count, uint32_t stride, uint8_t line_bits,
                           uint8_t t_bits) {
  return packing::packed_size_bytes(pair_count(count, stride),
                                    static_cast<uint8_t>(line_bits + t_bits));
}

/** Which pair holds value `index`, and whether it is the pair's x (0) or y (1). */
inline uint32_t pair_of(uint32_t index, uint32_t count, uint32_t stride,
                        uint32_t* component) {
  const uint32_t block = 2u * stride;
  const uint32_t full_end = (count / block) * block;
  if (index < full_end) {
    const uint32_t b = index / block;
    uint32_t r = index - b * block;
    *component = r >= stride ? 1u : 0u;
    if (r >= stride) r -= stride;
    return b * stride + r;
  }
  const uint32_t j = index - full_end;
  *component = j & 1u;
  return (full_end >> 1) + (j >> 1);
}

/** Decode pair `p` into (x, y). */
inline void decode_pair(const uint8_t* stream, const float* lines,
                        uint8_t line_bits, uint8_t t_bits, uint32_t p,
                        float* x, float* y) {
  const uint32_t code =
      packing::read_bits(stream, p, static_cast<uint8_t>(line_bits + t_bits));
  const uint32_t k = code & ((1u << line_bits) - 1u);
  const float tc = static_cast<float>(code >> line_bits);
  const float* l = lines + k * kLineFloats;
  *x = l[0] + tc * l[2];
  *y = l[1] + tc * l[3];
}

/** Decode all `count` values into `out` (e.g. once into RAM at start-up). */
inline void decode_all(const uint8_t* stream, const float* lines,
                       uint8_t line_bits, uint8_t t_bits, uint32_t stride,
                       uint32_t count, float* out) {
  const uint32_t block = 2u * stride;
  const uint32_t full = count / block;
  uint32_t p = 0;
  for (uint32_t b = 0; b < full; ++b) {
    float* o = out + b * block;
    for (uint32_t r = 0; r < stride; ++r, ++p) {
      decode_pair(stream, lines, line_bits, t_bits, p, o + r, o + r + stride);
    }
  }
  for (uint32_t i = full * block; i < count; i += 2, ++p) {
    float x, y;
    decode_pair(stream, lines, line_bits, t_bits, p, &x, &y);
    out[i] = x;
    if (i + 1 < count) out[i + 1] = y;
  }
}

}  // namespace hough
}  // namespace pTensor
#endif
