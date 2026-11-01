#ifndef PTENSOR_SQUEEZE_HPP
#define PTENSOR_SQUEEZE_HPP

#include "mathOpBase.hpp"

namespace pTensor {

template <typename T>
void squeeze_kernel(Tensor& in, const std::vector<uint8_t>& axis) {
  // Dimensions are uint16_t whatever the element type (they used to be
  // stored in T, so an int8 tensor lost any dimension above 127), and the
  // rank is updated.
  uint16_t dims[4] = {0, 0, 0, 0};
  TensorShape& shape = in->get_shape();
  const int rank = shape.num_dims();
  int dim_cursor = 0;
  for (int i = 0; i < rank; i++) {
    const bool drop =
        shape[i] == 1 &&
        (axis.empty() || std::find(axis.begin(), axis.end(), i) != axis.end());
    if (!drop) dims[dim_cursor++] = shape[i];
  }
  if (dim_cursor == 0) dims[dim_cursor++] = 1;  // squeezed to a scalar: [1]
  switch (dim_cursor) {
    case 1: shape = TensorShape(dims[0]); break;
    case 2: shape = TensorShape(dims[0], dims[1]); break;
    case 3: shape = TensorShape(dims[0], dims[1], dims[2]); break;
    default: shape = TensorShape(dims[0], dims[1], dims[2], dims[3]); break;
  }
}

namespace ReferenceOperators {

template <typename T>
class SqueezeOperator : public InPlaceFnc {
 public:
   SqueezeOperator() : _axis() {}
   SqueezeOperator(std::initializer_list<uint8_t> axis) : _axis(axis) {}

 protected:
  virtual void compute() { squeeze_kernel<T>(inputs[x].tensor(), _axis); }
 private:
  std::vector<uint8_t> _axis;
};
}
}

#endif
