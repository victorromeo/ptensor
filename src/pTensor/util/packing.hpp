#ifndef PTENSOR_UTIL_PACKING_HPP
#define PTENSOR_UTIL_PACKING_HPP
/**
 * Helpers for compact weight storage formats.
 *
 * Everything here is header-only and allocation-free, so it can run on the
 * device (decoding) as well as on the host (packing model weights offline).
 */
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pTensor {
namespace packing {

/** Number of bytes needed to store `count` values of `bits` bits each. */
inline size_t packed_size_bytes(size_t count, uint8_t bits) {
  return (count * bits + 7u) / 8u;
}

/**
 * Read value `index` from an LSB-first bit stream of `bits`-bit unsigned
 * values (1 <= bits <= 16). Value i occupies bits [i*bits, (i+1)*bits) of the
 * stream, where bit 0 is the least significant bit of byte 0.
 */
inline uint32_t read_bits(const uint8_t* data, uint32_t index, uint8_t bits) {
  const uint32_t bit_pos = index * bits;
  const uint8_t* p = data + (bit_pos >> 3);
  const uint32_t shift = bit_pos & 7u;
  // At most 16 + 7 = 23 bits are needed, i.e. three bytes.
  const uint32_t needed_bytes = (shift + bits + 7u) >> 3;
  uint32_t window = p[0];
  if (needed_bytes > 1) window |= static_cast<uint32_t>(p[1]) << 8;
  if (needed_bytes > 2) window |= static_cast<uint32_t>(p[2]) << 16;
  return (window >> shift) & ((1u << bits) - 1u);
}

/** Write value `index` into an LSB-first bit stream (see read_bits). */
inline void write_bits(uint8_t* data, uint32_t index, uint8_t bits,
                       uint32_t value) {
  const uint32_t mask = (1u << bits) - 1u;
  value &= mask;
  uint32_t bit_pos = index * bits;
  for (uint8_t written = 0; written < bits;) {
    uint8_t* p = data + (bit_pos >> 3);
    const uint32_t shift = bit_pos & 7u;
    const uint32_t room = 8u - shift;
    const uint32_t take = (bits - written) < room ? (bits - written) : room;
    const uint32_t chunk_mask = ((1u << take) - 1u) << shift;
    *p = static_cast<uint8_t>((*p & ~chunk_mask) |
                              (((value >> written) << shift) & chunk_mask));
    written = static_cast<uint8_t>(written + take);
    bit_pos += take;
  }
}

/** Pack `count` codes of `bits` bits each into `out` (packed_size_bytes long). */
inline void pack_bits(const uint32_t* codes, size_t count, uint8_t bits,
                      uint8_t* out) {
  memset(out, 0, packed_size_bytes(count, bits));
  for (size_t i = 0; i < count; ++i) {
    write_bits(out, static_cast<uint32_t>(i), bits, codes[i]);
  }
}

/** IEEE 754 binary16 -> binary32, including subnormals, infinities and NaN. */
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- 0
    } else {
      // Subnormal: normalise the mantissa.
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

/** IEEE 754 binary32 -> binary16 with round-to-nearest-even. */
inline uint16_t float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t abs = x & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u) {  // Inf or NaN
    return static_cast<uint16_t>(sign | 0x7C00u |
                                 (abs > 0x7F800000u ? 0x200u : 0u));
  }
  if (abs >= 0x477FF000u) {  // Rounds to >= 65520: overflow to infinity
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (abs < 0x38800000u) {  // Below the smallest normal half: subnormal or 0
    if (abs < 0x33000000u) return sign;  // Rounds to zero
    const uint32_t e = abs >> 23;
    const uint32_t m = (abs & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126u - e;  // 14 + (113 - e) - 1
    uint32_t half_m = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half_m & 1u))) ++half_m;
    return static_cast<uint16_t>(sign | half_m);
  }
  uint32_t h = ((abs >> 13) - ((127u - 15u) << 10));
  const uint32_t rem = abs & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(sign | h);
}

}  // namespace packing
}  // namespace pTensor
#endif
