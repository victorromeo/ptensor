#ifndef PTENSOR_SQUEEZE_HPP
#define PTENSOR_SQUEEZE_HPP

#include "mathOpBase.hpp"

namespace pTensor {

template <typename T>
void squeeze_kernel(Tensor& in, std::vector<uint8_t> axis) {
  T dims[4];
  memset(&dims, 0, 4*sizeof(T));
  TensorShape& shape = in->get_shape();
  // TODO optimize
  int dim_cursor = 0;
  for(int i = 0; i < 4; i++){
    if(shape[i] == 1) {
      if(axis.size() > 0){
        // Decide if we should keep or move on
        std::vector<uint8_t>::iterator it = std::find(axis.begin(), axis.end(), i);
        if(it == axis.end()){
          dims[dim_cursor] = shape[i];
          dim_cursor++;
        }
      }
    } else {
      dims[dim_cursor] = shape[i];
      dim_cursor++;
    }
  }
  for(int i = 0; i < 4; i++){
    shape[i] = dims[i];
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
