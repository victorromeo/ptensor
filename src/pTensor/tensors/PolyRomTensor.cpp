#include "PolyRomTensor.hpp"

#if PTENSOR_WITH_POLY_QUANT
#include "pTensor/util/packing.hpp"
#include "pTensor/util/polyquant.hpp"

namespace pTensor {

DEFINE_ERROR(InvalidPolyEncodingError);

size_t poly_stream_size(const PolyEncoding& e, uint32_t count) {
  if (!e.valid()) return 0;
  return polyquant::stream_bytes(count, e.bits);
}

PolyRomTensor::PolyRomTensor(const TensorShape& shape,
                             const PolyEncoding& encoding, const void* codes)
    : TensorInterface(shape, flt),
      _codes(reinterpret_cast<const uint8_t*>(codes)),
      _encoding(encoding),
      _slots{},
      _next_slot(0) {
  if (!encoding.valid() || codes == nullptr) {
    Context::get_default_context()->throwError(InvalidPolyEncodingError());
  }
}

PolyRomTensor::~PolyRomTensor() {}

void PolyRomTensor::resize(const TensorShape&) {
  Context::get_default_context()->throwError(InvalidResizeError());
}

float PolyRomTensor::decode(uint32_t i) const {
  const uint32_t k = packing::read_bits(_codes, i, _encoding.bits);
  if (_encoding.lut) return _encoding.lut[k];
  return polyquant::decode_code(_encoding.coeffs, _encoding.degree,
                                _encoding.bits, k);
}

void PolyRomTensor::decode_all(float* out) const {
  const uint32_t n = _shape.get_linear_size();
  for (uint32_t i = 0; i < n; ++i) out[i] = decode(i);
}

void* PolyRomTensor::read(uint32_t linear_index) const {
  float* slot = &_slots[_next_slot];
  _next_slot = static_cast<uint8_t>((_next_slot + 1u) % kDecodeSlots);
  *slot = decode(linear_index);
  return slot;
}

// Non-const element access goes through write() even for reads; decode as
// for read(). Anything written lands in a scratch slot and is discarded.
void* PolyRomTensor::write(uint32_t linear_index) {
  return read(linear_index);
}

}  // namespace pTensor
#endif  // PTENSOR_WITH_POLY_QUANT
