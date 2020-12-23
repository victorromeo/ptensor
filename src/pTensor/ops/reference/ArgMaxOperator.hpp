#ifndef PTENSOR_ARG_MAX_H
#define PTENSOR_ARG_MAX_H

#include "mathOpBase.hpp"

namespace pTensor {

namespace ReferenceOperators {

template <typename Tin>
class ArgMaxOperator : public OperatorInterface<2, 1>
{
public:
  enum names_in : uint8_t { input, axis};
  enum names_out : uint8_t { output };
protected:
  virtual void compute() {
    arg_min_max_kernel<Tin>(
      outputs[output].tensor(),
      inputs[input].tensor(),
      inputs[axis].tensor(),
      Max
    );
  }
};

}
} // namespace pTensor

#endif // PTENSOR_ARG_MIN_MAX_H
