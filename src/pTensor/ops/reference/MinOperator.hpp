#ifndef PTENSOR_MIN_HPP
#define PTENSOR_MIN_HPP

#include "mathOpBase.hpp"

namespace pTensor {

template <typename T>
T min_kernel(const Tensor& a) {
  T tmp = std::numeric_limits<T>::max();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for (uint32_t i = 0; i < a_size; i++) {
    tmp = std::min(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}

template <typename T>
void min_kernel(Tensor& out, const Tensor& a) {
  out(0) = min_kernel<T>(a);
}

namespace ReferenceOperators {

template <typename T>
class MinOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    min_kernel<T>(outputs[out].tensor(), inputs[in].tensor());
  }
};

}
}

#endif
