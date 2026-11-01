#ifndef PTENSOR_POLY_ROM_TENSOR_H
#define PTENSOR_POLY_ROM_TENSOR_H

#include "pTensor/ptensor_config.h"

#if PTENSOR_WITH_POLY_QUANT
#include "context.hpp"
#include "tensorBase.hpp"

namespace pTensor {

DECLARE_ERROR(InvalidPolyEncodingError);

/**
 * Parameters of a polynomially quantised tensor (see
 * pTensor/util/polyquant.hpp): `bits`-bit codes mapped to values by a degree
 * 1..3 polynomial, optionally through a lookup table.
 *
 * `lut`, if not null, holds 2^bits floats (the polynomial evaluated at every
 * code) and can live in flash or in RAM (polyquant::build_lut). If it is null
 * the polynomial is evaluated for each read.
 */
struct PolyEncoding {
  uint8_t degree;     // 1 linear, 2 quadratic, 3 cubic
  uint8_t bits;       // 1..16
  float coeffs[4];    // c0..c3; unused high terms are ignored
  const float* lut;   // optional, 2^bits floats

  static PolyEncoding make(uint8_t degree, uint8_t bits, const float* coeffs,
                           const float* lut = nullptr) {
    PolyEncoding e = {degree, bits, {0.0f, 0.0f, 0.0f, 0.0f}, lut};
    for (uint8_t i = 0; i <= degree && i < 4; ++i) e.coeffs[i] = coeffs[i];
    return e;
  }
  bool valid() const {
    return degree >= 1 && degree <= 3 && bits >= 1 && bits <= 16;
  }
};

/** Bytes of code stream needed for `count` values (coefficients/LUT not included). */
size_t poly_stream_size(const PolyEncoding& encoding, uint32_t count);

/**
 * A read-only float tensor stored as `bits`-bit codes that a quadratic or
 * cubic (or linear) polynomial maps to values, decoded on the fly through a
 * lookup table as operators read it. Use it for weights, like
 * PackedRomTensor; it changes no operator or other tensor type.
 *
 * The logical type is always flt. Writes are discarded.
 */
class PolyRomTensor : public TensorInterface {
 public:
  static constexpr uint8_t kDecodeSlots = 4;

  PolyRomTensor(const TensorShape& shape, const PolyEncoding& encoding,
                const void* codes);
  virtual ~PolyRomTensor();

  virtual void resize(const TensorShape& new_shape) override;
  const PolyEncoding& encoding() const { return _encoding; }

  /** Decode element `linear_index`. */
  float decode(uint32_t linear_index) const;
  /** Decode the whole tensor into `out` (num_elems() floats). */
  void decode_all(float* out) const;

 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;

 private:
  const uint8_t* _codes;
  PolyEncoding _encoding;
  mutable float _slots[kDecodeSlots];
  mutable uint8_t _next_slot;
};

}  // namespace pTensor
#endif  // PTENSOR_WITH_POLY_QUANT
#endif
