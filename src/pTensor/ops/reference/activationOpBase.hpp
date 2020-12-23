#ifndef PTENSOR_ACTIVATIONS_OPS_H
#define PTENSOR_ACTIVATIONS_OPS_H

#include <type_traits>

#include "pTensor/core/operatorBase.hpp"
#include <cmath>
#include <limits>
#include <functional>
#include <type_traits>

using std::exp;

namespace pTensor {

namespace Fuseable {

  template <typename T>
  using Activation = std::function<T(T)>;
  
  template <typename T>
  T NoActivation(T x) { return x; }
  
  template <typename T>
  T ReLU(T x) { return (x < 0) ? 0 : x; }
  
  template <typename T>
  T ReLU6(T x) { 
    if (x < 0){
      return 0;
    } else if (x > 6) {
      return 6;
    } else {
      return x;
    }
  }
  
  template <typename T>
  T Sigmoid(T x) {
    const T one = 1;
    return one / ( one + exp(-x) );
  }

} // namespace Fuseable

namespace ReferenceOperators {

class InPlaceActivationFnc : public OperatorInterface<1, 0> {
 public:
  enum names_in : uint8_t { x };

 protected:
  virtual void compute() = 0;
};

} 

}  // namespace pTensor

#endif
