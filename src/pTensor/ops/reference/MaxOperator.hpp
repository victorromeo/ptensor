#ifndef PTENSOR_MAX_HPP
#define PTENSOR_MAX_HPP

#include "mathOpBase.hpp"

namespace pTensor {

template <typename T>
T max_kernel(const Tensor& a) {
  T tmp = std::numeric_limits<T>::lowest();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for (uint32_t i = 0; i < a_size; i++) {
    tmp = std::max(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}

template <typename T>
void max_kernel(Tensor& out, const Tensor& a) {
  out(0) = max_kernel<T>(a);
}

namespace ReferenceOperators {

template <typename T>
class MaxOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    max_kernel<T>(outputs[out].tensor(), inputs[in].tensor());
  }
};

}
}

#endif
