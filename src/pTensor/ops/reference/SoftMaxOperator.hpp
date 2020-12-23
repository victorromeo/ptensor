#ifndef PTENSOR_ACTIVATIONS_OPS_SOFTMAX_H
#define PTENSOR_ACTIVATIONS_OPS_SOFTMAX_H
#include <type_traits>

#include "activationOpBase.hpp"

namespace pTensor {

template <typename T>
void inplace_softmax_k(Tensor& in, T beta = 1) {
  T tmp;
  T mSum = 0;
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(i, j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(i,j)) - max) * beta);
      mSum += tmp;
      in(i,j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      in(i, j)  = static_cast<T>(in(i, j)) / mSum;
    }
  }
}

template <typename T>
void softmax_k(Tensor& out, const Tensor& in, T beta=1) {
  T tmp;
  T mSum = 0;
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(i, j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(i,j)) - max) * beta);
      mSum += tmp;
      out(i,j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      out(i, j)  = static_cast<T>(out(i, j)) / mSum;
    }
  }

}

void sq_softmax_k(Tensor& out, const Tensor& in, int8_t beta) {
  const float beta_f = static_cast<float>(beta);
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = static_cast<float>(std::numeric_limits<int8_t>::lowest());
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<float>(static_cast<int8_t>(in(i, j))));
    }

    float mSum = 0;
    for(int j = 0; j < depth; j++){
      const int32_t in32 =  static_cast<int8_t>(in(i,j));
      const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
      const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
      const float in_f = (in32 - in_zp)*in_scale;
      const float tmp = exp((in_f - max) * beta_f);
      mSum += tmp;
      //out(i,j) = tmp;
    }
    // TODO FIXME SLOW but mem efficient
    for(int j = 0; j < depth; j++){
      const int32_t in32 =  static_cast<int8_t>(in(i,j));
      const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
      const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
      const float in_f = (in32 - in_zp)*in_scale;
      const float out_val = exp((in_f - max) * beta_f) / mSum;
      
      const float oscale = out->get_quantization_params().get_scale_for_channel(0);
      const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);
      const int32_t otmp = static_cast<int32_t>(out_val/oscale) + ozp;
      const int8_t out8 = (otmp < -127 ) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);
      
      out(i, j)  = out8;
    }
  }

}

namespace ReferenceOperators {

template <typename T>
class InPlaceSoftmax : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct Softmax on non-signed types");

 public:
  InPlaceSoftmax() : beta(1) {}
  InPlaceSoftmax(T beta) : beta(beta) {}
 protected:
  virtual void compute() { inplace_softmax_k<T>(inputs[x].tensor(), beta); }

 private:
  T beta;
};

template <typename T>
class SoftmaxOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct softmax on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 public:
  SoftmaxOperator() : beta(1) {}
  SoftmaxOperator(T beta) : beta(beta) {}
 protected:
  virtual void compute();

 private:
  T beta;
};

template <typename T>
void SoftmaxOperator<T>::compute() {
  const Tensor& inT = inputs[in].tensor();
  Tensor& outT = outputs[out].tensor();
  // TODO Check sizes here and throw mismatch
  uint32_t in_size = inT->get_shape().get_linear_size();
  uint32_t out_size = outT->get_shape().get_linear_size();
  if (in_size != out_size)
    Context::get_default_context()->throwError(
        new OperatorIOSizeMismatchError);
  softmax_k<T>(outT, inT, beta);
}

// Symmetric Quantized reference
template <>
void SoftmaxOperator<int8_t>::compute() {
  const Tensor& inT = inputs[in].tensor();
  Tensor& outT = outputs[out].tensor();
  // TODO Check sizes here and throw mismatch
  uint32_t in_size = inT->get_shape().get_linear_size();
  uint32_t out_size = outT->get_shape().get_linear_size();
  if (in_size != out_size)
    Context::get_default_context()->throwError(
        new OperatorIOSizeMismatchError);
  sq_softmax_k(outT, inT, beta);
}

} 
}  // namespace pTensor

#endif
