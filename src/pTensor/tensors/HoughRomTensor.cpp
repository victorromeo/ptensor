#include "HoughRomTensor.hpp"

#if PTENSOR_WITH_HOUGH
#include "pTensor/util/hough.hpp"

namespace pTensor {

DEFINE_ERROR(InvalidHoughEncodingError);

size_t hough_stream_size(const HoughEncoding& e, uint32_t count) {
  if (!e.valid()) return 0;
  return hough::stream_bytes(count, e.pair_stride, e.line_bits, e.t_bits);
}

HoughRomTensor::HoughRomTensor(const TensorShape& shape,
                               const HoughEncoding& encoding,
                               const void* stream)
    : TensorInterface(shape, flt),
      _stream(reinterpret_cast<const uint8_t*>(stream)),
      _encoding(encoding),
      _count(shape.get_linear_size()),
      _cached_pair(0xFFFFFFFFu),
      _cached{0.0f, 0.0f},
      _slots{},
      _next_slot(0) {
  if (!encoding.valid() || stream == nullptr) {
    Context::get_default_context()->throwError(InvalidHoughEncodingError());
  }
}

HoughRomTensor::~HoughRomTensor() {}

void HoughRomTensor::resize(const TensorShape&) {
  Context::get_default_context()->throwError(InvalidResizeError());
}

float HoughRomTensor::decode(uint32_t i) const {
  uint32_t component;
  const uint32_t p = hough::pair_of(i, _count, _encoding.pair_stride, &component);
  if (p != _cached_pair) {
    hough::decode_pair(_stream, _encoding.lines, _encoding.line_bits,
                       _encoding.t_bits, p, &_cached[0], &_cached[1]);
    _cached_pair = p;
  }
  return _cached[component];
}

void HoughRomTensor::decode_all(float* out) const {
  hough::decode_all(_stream, _encoding.lines, _encoding.line_bits,
                    _encoding.t_bits, _encoding.pair_stride, _count, out);
}

void* HoughRomTensor::read(uint32_t linear_index) const {
  float* slot = &_slots[_next_slot];
  _next_slot = static_cast<uint8_t>((_next_slot + 1u) % kDecodeSlots);
  *slot = decode(linear_index);
  return slot;
}

// Non-const element access goes through write() even for reads; decode as
// for read(). Anything written lands in a scratch slot and is discarded.
void* HoughRomTensor::write(uint32_t linear_index) {
  return read(linear_index);
}

}  // namespace pTensor
#endif  // PTENSOR_WITH_HOUGH
