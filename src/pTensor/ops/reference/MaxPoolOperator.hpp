#ifndef PTENSOR_POOLING_MAX_H
#define PTENSOR_POOLING_MAX_H

#include "genericPoolOpBase.hpp"

namespace pTensor {

template <typename T>
class MaxFilter {
  T tmp;
  int16_t h;
  int16_t w;
  int16_t c;

 public:
  MaxFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = std::numeric_limits<T>::lowest(); }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp = std::max(tmp, input_value);
  }
  inline T finalize() const { return tmp; }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

namespace ReferenceOperators {

template <typename T>
using MaxPoolOperator = GenericPoolOperator<T, MaxFilter<T>>;

}

}

#endif