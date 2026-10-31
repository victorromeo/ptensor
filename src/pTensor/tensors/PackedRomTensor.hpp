#ifndef PTENSOR_PACKED_ROM_TENSOR_H
#define PTENSOR_PACKED_ROM_TENSOR_H

#include "context.hpp"
#include "tensorBase.hpp"

namespace pTensor {

DECLARE_ERROR(InvalidPackedFormatError);

/**
 * How the bytes of a PackedRomTensor are laid out in ROM/flash.
 *
 * All multi-byte formats are little endian and need no alignment.
 */
enum class PackedFormat : uint8_t {
  UIntBits,  // Unsigned integers of 1..16 bits, LSB-first bit stream
  Int8,
  UInt8,
  Int16,
  UInt16,
  Int32,
  Float16,  // IEEE 754 binary16
  Float32,  // IEEE 754 binary32
  Float64,  // IEEE 754 binary64
};

/**
 * Describes a packed buffer and how to turn each stored code into a value.
 *
 * Integer codes are mapped to real values with an affine transform,
 *   real = (code - zero_point) * scale,
 * and floating point formats are used as-is (scale/zero_point are ignored).
 * The real value is then converted to the tensor's logical element type:
 * float tensors receive it directly, integer tensors receive it rounded to
 * nearest and saturated to the type's range.
 *
 * Example: 6-bit weights in flash, read by ops as float:
 *   PackedEncoding::bits(6, 0.02f, 32)
 */
struct PackedEncoding {
  PackedFormat format;
  uint8_t bit_width;  // Only used for UIntBits (1..16)
  float scale;
  int32_t zero_point;

  static PackedEncoding bits(uint8_t nbits, float scale = 1.0f,
                             int32_t zero_point = 0) {
    PackedEncoding e = {PackedFormat::UIntBits, nbits, scale, zero_point};
    return e;
  }
  static PackedEncoding of(PackedFormat f, float scale = 1.0f,
                           int32_t zero_point = 0) {
    PackedEncoding e = {f, 0, scale, zero_point};
    return e;
  }
};

/** Bytes needed to store `count` elements in `encoding`. */
size_t packed_buffer_size(const PackedEncoding& encoding, uint32_t count);

/**
 * A read-only tensor whose data stays compressed in ROM/flash and is decoded
 * element by element as operators read it. Use it for model weights to trade a
 * little compute for a large cut in storage, e.g. 6- or 7-bit weights instead
 * of 8-bit, or fp16/fp64 constants presented to float kernels.
 *
 * The tensor's logical type (what operators see) can be any pTensor ttype.
 * Quantization parameters set on the tensor describe that logical type, just
 * as for any other tensor.
 *
 * Writes are discarded: the packed data is never modified.
 *
 * Decoded values live in a small ring of scratch slots, so up to
 * kDecodeSlots values read from the same tensor may be held unconverted at
 * once (reference kernels convert each read immediately).
 */
class PackedRomTensor : public TensorInterface {
 public:
  static constexpr uint8_t kDecodeSlots = 4;

  PackedRomTensor(const TensorShape& shape, ttype logical_type,
                  const PackedEncoding& encoding, const void* buffer);
  virtual ~PackedRomTensor();

  virtual void resize(const TensorShape& new_shape) override;

  const PackedEncoding& encoding() const { return _encoding; }
  // Decode element `linear_index` to its real value (before conversion to the
  // logical type). Handy for tests and debugging.
  float decode_real(uint32_t linear_index) const;

 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;

 private:
  const uint8_t* _data;
  PackedEncoding _encoding;
  // Decode scratch. 4-byte elements cover every logical ttype.
  mutable uint32_t _slots[kDecodeSlots];
  mutable uint8_t _next_slot;
};

}  // namespace pTensor
#endif
