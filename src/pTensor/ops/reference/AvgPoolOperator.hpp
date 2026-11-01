#ifndef PTENSOR_POOLING_AVG_H
#define PTENSOR_POOLING_AVG_H

#include <type_traits>

#include "genericPoolOpBase.hpp"

namespace pTensor {

/**
 * Mean of the taps inside the image (padding is not counted, as in
 * TensorFlow). Integer types accumulate in int32_t and round half away from
 * zero like TFLite's int8 AVERAGE_POOL_2D (they used to sum in the element
 * type, so an int8 window overflowed, and truncate).
 */
template <typename T>
class AvgFilter {
  typedef typename std::conditional<std::is_integral<T>::value, int32_t, T>::type Acc;
  Acc tmp;
  int32_t count;
  int16_t h;
  int16_t w;
  int16_t c;

  template <typename U = T>
  typename std::enable_if<std::is_integral<U>::value, T>::type mean() const {
    if (count == 0) return 0;
    const int32_t half = count / 2;
    return static_cast<T>(tmp > 0 ? (tmp + half) / count : (tmp - half) / count);
  }
  template <typename U = T>
  typename std::enable_if<!std::is_integral<U>::value, T>::type mean() const {
    return count ? static_cast<T>(tmp / static_cast<T>(count)) : T(0);
  }

 public:
  AvgFilter(int16_t h, int16_t w, int16_t c) : tmp(0), count(0), h(h), w(w), c(c) {}
  inline void reset() { tmp = 0; count = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp += static_cast<Acc>(input_value);
    ++count;
  }
  inline T finalize() const { return mean(); }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

namespace ReferenceOperators {

template <typename T>
using AvgPoolOperator = GenericPoolOperator<T, AvgFilter<T>>;

}

}

#endif