#ifndef PTENSOR_ARITH_ADD_H
#define PTENSOR_ARITH_ADD_H

#include "mathOpBase.hpp"
#include "pTensor/core/fast_access.hpp"

namespace pTensor {

/*add_kernel : performs the add operation logic*/
template <typename T>
void add_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // Decide on c shape
  TensorShape c_shape = c->get_shape();
  uint32_t c_size = c_shape.get_linear_size();
  // TensorInterface& C = reinterpret_cast<TensorInterface*>(*c);
  // const TensorInterface& A = reinterpret_cast<TensorInterface*>(*a);
  // const TensorInterface& B = reinterpret_cast<TensorInterface*>(*b);

  {
    const T* pa = fast_read<T>(a);
    const T* pb = fast_read<T>(b);
    T* pc = fast_write<T>(c);
    if (pa && pb && pc) {
      for (uint32_t i = 0; i < c_size; i++)
        pc[i] = static_cast<T>(pa[i] + pb[i]);
      return;
    }
  }
  for (uint32_t i = 0; i < c_size; i++)
    c(i) = static_cast<T>(static_cast<T>(a(i)) + static_cast<T>(b(i)));
}

namespace ReferenceOperators {

/* AddOperator : Adds tensors a and b, resulting in c output tensor */
template <typename T>
class AddOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    add_kernel<T>(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

}

}  // namespace pTensor
#endif