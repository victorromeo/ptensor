#ifndef PTENSOR_REDUCE_OPS_H
#define PTENSOR_REDUCE_OPS_H

#include <cmath>
#include <initializer_list>
#include <type_traits>

#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {

namespace ReferenceOperators {

/**
 * Mean over a set of axes (TFLite MEAN / numpy.mean with keepdims either way:
 * the output holds the kept dimensions in order, so [2,3,4] reduced over
 * axis 1 gives 2 * 4 values, whether the output is shaped [2,4] or [2,1,4]).
 *
 * Sums are taken in double. Integer element types are averaged as raw values
 * and rounded half away from zero (input and output must share their
 * quantization).
 */
template <typename T>
class ReduceMeanOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };

  ReduceMeanOperator(std::initializer_list<uint16_t> axes) : _mask(0) {
    for (uint16_t a : axes) add_axis(a);
  }
  ReduceMeanOperator(const uint16_t* axes, uint8_t num_axes) : _mask(0) {
    for (uint8_t i = 0; i < num_axes; ++i) add_axis(axes[i]);
  }

 protected:
  virtual void compute() {
    const Tensor& in = inputs[input].tensor();
    Tensor& out = outputs[output].tensor();
    const TensorShape& s = in->get_shape();
    const int rank = s.num_dims();
    uint32_t kept[4], red[4], kept_stride[4], red_stride[4];
    int nk = 0, nr = 0;
    uint32_t stride = 1, osz = 1, rsz = 1;
    uint32_t strides[4];
    for (int i = rank - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= s[i];
    }
    for (int i = 0; i < rank; ++i) {
      if (_mask & (1u << i)) {
        red[nr] = s[i];
        red_stride[nr++] = strides[i];
        rsz *= s[i];
      } else {
        kept[nk] = s[i];
        kept_stride[nk++] = strides[i];
        osz *= s[i];
      }
    }
    if ((_mask >> rank) != 0 || out->get_shape().get_linear_size() != osz) {
      Context::get_default_context()->throwError(InvalidTensorDimensionsError());
      return;
    }
    for (uint32_t o = 0; o < osz; ++o) {
      uint32_t base = 0, rem = o;
      for (int k = nk - 1; k >= 0; --k) {
        base += (rem % kept[k]) * kept_stride[k];
        rem /= kept[k];
      }
      double acc = 0.0;
      for (uint32_t r = 0; r < rsz; ++r) {
        uint32_t off = base, rr = r;
        for (int k = nr - 1; k >= 0; --k) {
          off += (rr % red[k]) * red_stride[k];
          rr /= red[k];
        }
        acc += static_cast<double>(static_cast<T>(in(off)));
      }
      const double mean = acc / rsz;
      out(o) = std::is_integral<T>::value ? static_cast<T>(std::round(mean))
                                          : static_cast<T>(mean);
    }
  }

 private:
  void add_axis(uint16_t a) { _mask |= (a < 4) ? (1u << a) : (1u << 4); }
  uint32_t _mask;
};

}  // namespace ReferenceOperators
}  // namespace pTensor
#endif
