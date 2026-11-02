#ifndef PTENSOR_SLICE_OPS_H
#define PTENSOR_SLICE_OPS_H
/**
 * Strided slice (NumPy x[b0::s0, b1::s1, ...], PyTorch narrow / select /
 * chunk / split / x[..., a:b:s]): output element (i0, i1, ...) is input
 * element (b0 + i0 * s0, b1 + i1 * s1, ...). The output tensor's shape gives
 * how many elements each dimension keeps; begin and step are per input
 * dimension (steps are positive). Input and output have the same rank.
 */
#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/core/fast_access.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {

DECLARE_ERROR(InvalidSliceError);

namespace ReferenceOperators {

template <typename T>
class SliceOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  SliceOperator(const uint16_t* begin, const uint16_t* step, uint8_t rank) : _rank(rank) {
    for (int i = 0; i < 4; ++i) {
      _begin[i] = (i < rank) ? begin[i] : 0;
      _step[i] = (i < rank) ? step[i] : 1;
    }
  }

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    Tensor& y = outputs[out].tensor();
    const TensorShape& is = x->get_shape();
    const TensorShape& os = y->get_shape();
    const int r = _rank;
    if (r < 1 || r > 4 || is.num_dims() != r || os.num_dims() != r) {
      Context::get_default_context()->throwError(InvalidSliceError());
      return;
    }
    // Right-align to 4 dims: dims of size 1 with begin 0, step 1.
    uint32_t idim[4], odim[4], begin[4], step[4], istride[4];
    for (int d = 0; d < 4; ++d) {
      const int k = d - (4 - r);
      idim[d] = k >= 0 ? is[k] : 1;
      odim[d] = k >= 0 ? os[k] : 1;
      begin[d] = k >= 0 ? _begin[k] : 0;
      step[d] = k >= 0 ? _step[k] : 1;
      if (step[d] == 0 || (odim[d] > 0 && begin[d] + (odim[d] - 1) * step[d] >= idim[d])) {
        Context::get_default_context()->throwError(InvalidSliceError());
        return;
      }
    }
    uint32_t st = 1;
    for (int d = 3; d >= 0; --d) {
      istride[d] = st;
      st *= idim[d];
    }
    const T* px = fast_read<T>(x);
    T* py = fast_write<T>(y);
    uint32_t o = 0;
    for (uint32_t a = 0; a < odim[0]; ++a)
      for (uint32_t b = 0; b < odim[1]; ++b)
        for (uint32_t c = 0; c < odim[2]; ++c) {
          uint32_t base = (begin[0] + a * step[0]) * istride[0] +
                          (begin[1] + b * step[1]) * istride[1] +
                          (begin[2] + c * step[2]) * istride[2] + begin[3];
          for (uint32_t e = 0; e < odim[3]; ++e, ++o) {
            const uint32_t i = base + e * step[3];
            if (px && py) {
              py[o] = px[i];
            } else {
              const T v = x(i);
              y(o) = v;
            }
          }
        }
  }

 private:
  uint8_t _rank;
  uint16_t _begin[4], _step[4];
};

}  // namespace ReferenceOperators
}  // namespace pTensor
#endif
