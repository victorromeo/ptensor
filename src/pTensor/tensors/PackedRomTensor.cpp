#include "PackedRomTensor.hpp"

#include <cmath>
#include <cstring>

#include "pTensor/util/packing.hpp"

namespace pTensor {

DEFINE_ERROR(InvalidPackedFormatError);

namespace {

inline uint16_t load_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
inline uint32_t load_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

template <typename T>
inline T saturate_round(float v, float lo, float hi) {
  if (!(v == v)) return 0;  // NaN
  float r = v < 0 ? v - 0.5f : v + 0.5f;  // round half away from zero
  if (r <= lo) return static_cast<T>(lo);
  if (r >= hi) return static_cast<T>(hi);
  return static_cast<T>(r);
}

bool encoding_is_valid(const PackedEncoding& e) {
  if (e.format == PackedFormat::UIntBits) return e.bit_width >= 1 && e.bit_width <= 16;
  return e.format <= PackedFormat::Float64;
}

}  // namespace

size_t packed_buffer_size(const PackedEncoding& e, uint32_t count) {
  switch (e.format) {
    case PackedFormat::UIntBits:
      return packing::packed_size_bytes(count, e.bit_width);
    case PackedFormat::Int8:
    case PackedFormat::UInt8:
      return count;
    case PackedFormat::Int16:
    case PackedFormat::UInt16:
    case PackedFormat::Float16:
      return 2u * count;
    case PackedFormat::Int32:
    case PackedFormat::Float32:
      return 4u * count;
    case PackedFormat::Float64:
      return 8u * count;
  }
  return 0;
}

PackedRomTensor::PackedRomTensor(const TensorShape& shape, ttype logical_type,
                                 const PackedEncoding& encoding,
                                 const void* buffer)
    : TensorInterface(shape, logical_type),
      _data(reinterpret_cast<const uint8_t*>(buffer)),
      _encoding(encoding),
      _slots{},
      _next_slot(0) {
  if (!encoding_is_valid(encoding) || logical_type == undefined) {
    Context::get_default_context()->throwError(InvalidPackedFormatError());
  }
}

PackedRomTensor::~PackedRomTensor() {}

void PackedRomTensor::resize(const TensorShape& new_shape) {
  Context::get_default_context()->throwError(InvalidResizeError());
}

// Non-const element access goes through write() even when the caller only
// reads (e.g. `t(i)` on a non-const Tensor handle), so decode as for read().
// Any value actually written lands in a scratch slot and is discarded; the
// packed data in ROM is never modified.
void* PackedRomTensor::write(uint32_t linear_index) {
  return read(linear_index);
}

float PackedRomTensor::decode_real(uint32_t i) const {
  const PackedEncoding& e = _encoding;
  int32_t code;
  switch (e.format) {
    case PackedFormat::UIntBits:
      code = static_cast<int32_t>(packing::read_bits(_data, i, e.bit_width));
      break;
    case PackedFormat::Int8:
      code = static_cast<int8_t>(_data[i]);
      break;
    case PackedFormat::UInt8:
      code = _data[i];
      break;
    case PackedFormat::Int16:
      code = static_cast<int16_t>(load_u16(_data + 2u * i));
      break;
    case PackedFormat::UInt16:
      code = load_u16(_data + 2u * i);
      break;
    case PackedFormat::Int32:
      code = static_cast<int32_t>(load_u32(_data + 4u * i));
      break;
    case PackedFormat::Float16:
      return packing::half_to_float(load_u16(_data + 2u * i));
    case PackedFormat::Float32: {
      const uint32_t b = load_u32(_data + 4u * i);
      float f;
      memcpy(&f, &b, sizeof(f));
      return f;
    }
    case PackedFormat::Float64: {
      double d;
      memcpy(&d, _data + 8u * i, sizeof(d));
      return static_cast<float>(d);
    }
    default:
      return 0.0f;
  }
  return (static_cast<float>(code) - static_cast<float>(e.zero_point)) * e.scale;
}

void* PackedRomTensor::read(uint32_t linear_index) const {
  uint32_t* slot = &_slots[_next_slot];
  _next_slot = static_cast<uint8_t>((_next_slot + 1u) % kDecodeSlots);

  const float real = decode_real(linear_index);
  switch (_type) {
    case flt: {
      memcpy(slot, &real, sizeof(real));
      break;
    }
    case i8: {
      int8_t v = saturate_round<int8_t>(real, -128.0f, 127.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    case u8: {
      uint8_t v = saturate_round<uint8_t>(real, 0.0f, 255.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    case i16: {
      int16_t v = saturate_round<int16_t>(real, -32768.0f, 32767.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    case u16: {
      uint16_t v = saturate_round<uint16_t>(real, 0.0f, 65535.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    case i32: {
      int32_t v = saturate_round<int32_t>(real, -2147483648.0f, 2147483520.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    case u32: {
      uint32_t v = saturate_round<uint32_t>(real, 0.0f, 4294967040.0f);
      memcpy(slot, &v, sizeof(v));
      break;
    }
    default:
      *slot = 0;
      break;
  }
  return slot;
}

}  // namespace pTensor
