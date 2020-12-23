#ifndef PTENSOR_RESHAPE_H
#define PTENSOR_RESHAPE_H

#include "pTensor/core/context.hpp"
#include "pTensor/core/types.hpp"
#include "pTensor/core/tensor.hpp"
#include "pTensor/core/pTensor_util.hpp"
#include "pTensor/core/operatorBase.hpp"

using std::array;

namespace pTensor {
namespace ReferenceOperators {

template <typename Tin>
class ReshapeOperator : public OperatorInterface<1, 1> {
/* reshape input as the shape of output*/
public:
  ReshapeOperator(const TensorShape&& shape) : _shape(shape) {}
  ReshapeOperator(const TensorShape& shape) : _shape(shape) {}
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };
  virtual void compute(){
    const Tensor& input_tensor = inputs[input].tensor();
    Tensor& output_tensor = outputs[output].tensor();
    output_tensor->resize(_shape);
    if (input_tensor->num_elems() != output_tensor->num_elems()){
      pTensor_printf("inconsistent input and output shape for reshape\n");
      Context::get_default_context()->throwError(new InvalidReshapeError);
      return;
    }
    if (input_tensor->get_type() != output_tensor->get_type()){
      pTensor_printf("inconsistent input and output data type for reshape\n");
      Context::get_default_context()->throwError(new InvalidTensorDataTypeError);
      return;
    }
    if (!_check_input_shape()){
      Context::get_default_context()->throwError(new InvalidTensorDataTypeError);
      return;
    }
    // copy data
    for (uint32_t i = 0; i < input_tensor->num_elems(); ++i) { 
      // this is not copy: `output_tensor(i) = input_tensor(i);`
      output_tensor(i) = static_cast<Tin>(input_tensor(i));
    }
  }
private:
  TensorShape _shape;
  bool _check_input_shape(){
    const Tensor& input_tensor = inputs[input].tensor();
    const TensorShape& shape = input_tensor->get_shape();
    uint8_t num_dims = shape.num_dims();
    for (int i = 0; i < num_dims; ++i){
      if (shape[i] < 0) {
        pTensor_printf("the output shape must be all positive\n");
        return false;
      }
    }
    return true;
  }
};

}
}

#endif // PTENSOR_RESHAPE_H
