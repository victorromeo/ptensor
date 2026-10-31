#ifndef PTENSOR_HOUGH_ROM_TENSOR_H
#define PTENSOR_HOUGH_ROM_TENSOR_H

#include "pTensor/ptensor_config.h"

#if PTENSOR_WITH_HOUGH
#include "context.hpp"
#include "tensorBase.hpp"

namespace pTensor {

DECLARE_ERROR(InvalidHoughEncodingError);

/**
 * Parameters of a Hough line-codebook tensor (see pTensor/util/hough.hpp for
 * the format). `lines` points at 4 * 2^line_bits floats (ax, ay, dx, dy per
 * line) and, like the code stream, can live in ROM/flash.
 */
struct HoughEncoding {
  uint8_t line_bits;     // 0..8: the codebook has 2^line_bits lines
  uint8_t t_bits;        // 1..16 - line_bits: positions along a line
  uint16_t pair_stride;  // values i and i + pair_stride form a pair
  const float* lines;

  static HoughEncoding make(uint8_t line_bits, uint8_t t_bits,
                            uint16_t pair_stride, const float* lines) {
    HoughEncoding e = {line_bits, t_bits, pair_stride, lines};
    return e;
  }
  bool valid() const {
    return lines != nullptr && line_bits <= 8 && t_bits >= 1 &&
           line_bits + t_bits <= 16 && pair_stride >= 1;
  }
};

/** Bytes of code stream needed for `count` values (codebook not included). */
size_t hough_stream_size(const HoughEncoding& encoding, uint32_t count);

/**
 * A read-only float tensor stored as a Hough line codebook plus a bit stream
 * of (line, position) codes, decoded on the fly as operators read it. Use it
 * for weights, like PackedRomTensor; it does not change any operator, so
 * existing tensor types and kernels run exactly as before.
 *
 * The logical type is always flt. Writes are discarded.
 */
class HoughRomTensor : public TensorInterface {
 public:
  static constexpr uint8_t kDecodeSlots = 4;

  HoughRomTensor(const TensorShape& shape, const HoughEncoding& encoding,
                 const void* stream);
  virtual ~HoughRomTensor();

  virtual void resize(const TensorShape& new_shape) override;
  const HoughEncoding& encoding() const { return _encoding; }

  /** Decode element `linear_index`. */
  float decode(uint32_t linear_index) const;
  /** Decode the whole tensor into `out` (num_elems() floats). */
  void decode_all(float* out) const;

 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;

 private:
  const uint8_t* _stream;
  HoughEncoding _encoding;
  uint32_t _count;
  // The last decoded pair: kernels often read both halves of a pair close
  // together, and each decode yields two values.
  mutable uint32_t _cached_pair;
  mutable float _cached[2];
  mutable float _slots[kDecodeSlots];
  mutable uint8_t _next_slot;
};

}  // namespace pTensor
#endif  // PTENSOR_WITH_HOUGH
#endif
