#ifndef PTENSOR_ACTIVATIONS_OPS_LOGISTIC_H
#define PTENSOR_ACTIVATIONS_OPS_LOGISTIC_H

#include "SoftMaxOperator.hpp"
#include "SigmoidOperator.hpp"

namespace pTensor {

namespace ReferenceOperators {

template <typename T>
class InPlaceLogistic : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Logistic on non-signed types");

 protected:
  virtual void compute();
};
  
template <typename T>
void InPlaceLogistic<T>::compute() { inplace_softmax_k<T>(inputs[x].tensor()); }


template <typename T>
class LogisticOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Logistic on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute();
};

template <typename T>
void LogisticOperator<T>::compute() {
  const Tensor& inT = inputs[in].tensor();
  Tensor& outT = outputs[out].tensor();
  // TODO Check sizes here and throw mismatch
  uint32_t in_size = inT->get_shape().get_linear_size();
  uint32_t out_size = outT->get_shape().get_linear_size();
  if (in_size != out_size)
    Context::get_default_context()->throwError(
        new OperatorIOSizeMismatchError);
  sigmoid_k<T>()(outT, inT);
}

} 
}  // namespace pTensor

#endif
