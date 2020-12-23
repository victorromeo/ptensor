#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {

namespace ReferenceOperators {

// https://github.com/tensorflow/tensorflow/blob/12a806e96866296b154134b27ef4228f39f403cc/tensorflow/lite/micro/kernels/reduce.cc#L83
class ReduceOperator : public OperatorInterface<2, 1> {
 public:
  ReduceOperator(initializer_list<uint16_t> dims);
  uint32_t adjust_linear_idx(Tensor& tensor, uint32_t idx);

 private:
  uint16_t _dims[4];
};

template <typename T>
class ReduceMeanOperator : ReduceOperator {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };
  ReduceMeanOperator(initializer_list<uint32_t> dims) : ReduceOperator(dims) {}

 protected:
  void compute() {
    Tensor& inputT = inputs[input].tensor();
    Tensor& outputT = outputs[output].tensor();
    for (uint32_t i = 0; i < outputT->num_elems(); ++i) {
      outputT(i) = static_cast<T>(0);
    }
    T denum = 1;
    for (auto d : _dims) {
      denum *= inputT->get_shape()[d];
    }
    for (uint32_t offset = 0; offset < inputT->num_elems(); ++offset) {
      uint32_t new_offset = adjust_linear_idx(input, offset);
      T value = inputT(offset) / denum;
      outputT(new_offset) += value;
    }
  }
};

ReduceOperator::ReduceOperator(initializer_list<uint16_t> dims) {
  size_t i = 0;
  for (auto s : dims) {
    _dims[i++] = s;
  }
}
uint32_t ReduceOperator::adjust_linear_idx(Tensor& tensor, uint32_t idx) {
  TensorShape ori_shape = tensor->get_shape();
  TensorStrides ori_strides = TensorStrides(ori_shape);
  TensorShape reduced_shape = TensorShape(0, 0, 0, 0);
  int c = 0;
  for (int i = 0; i < 4; ++i) {
    bool is_reduce_dim = false;
    for (size_t j = 0; j < 4; ++j) {
      if (i == _dims[j]) {
        is_reduce_dim = true;
        break;
      }
    }
    if (!is_reduce_dim) {
      reduced_shape[c] = ori_shape[i];
      c++;
    }
  }
  reduced_shape.update_dims();
  TensorStrides reduced_strides = TensorStrides(reduced_shape);
  uint32_t new_idx = 0;
  size_t current_idx = 0;
  uint32_t residual = idx;
  for (size_t i = 0; i < ori_shape.num_dims(); ++i) {
    uint32_t axis_size = ori_shape[i];
    uint32_t stride = ori_strides[i];
    uint32_t q = std::min(residual / stride, axis_size - 1);
    bool is_reduce_dim = false;
    for (auto d : _dims) {
      if (d == i) {
        is_reduce_dim = true;
        break;
      }
    }
    if (!is_reduce_dim) {
      new_idx += reduced_strides[current_idx] * q;
      current_idx++;
    }
    residual -= q * stride;
  }
  return new_idx;
}

}  // namespace ReferenceOperators
}  // namespace pTensor
