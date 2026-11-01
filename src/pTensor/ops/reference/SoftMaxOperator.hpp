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
    // Linear indexing: the softmax runs over the last axis of any rank (the
    // 2D in(i, j) form indexed rank 3 and 4 tensors out of bounds).
    const uint32_t row = static_cast<uint32_t>(i) * depth;
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(row + j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(row + j)) - max) * beta);
      mSum += tmp;
      in(row + j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      in(row + j)  = static_cast<T>(in(row + j)) / mSum;
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
    // Linear indexing: the softmax runs over the last axis of any rank (the
    // 2D in(i, j) form indexed rank 3 and 4 tensors out of bounds).
    const uint32_t row = static_cast<uint32_t>(i) * depth;
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(row + j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(row + j)) - max) * beta);
      mSum += tmp;
      out(row + j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      out(row + j)  = static_cast<T>(out(row + j)) / mSum;
    }
  }

}

inline void sq_softmax_k(Tensor& out, const Tensor& in, int8_t beta) {
  // Dequantize, float softmax, requantize (round to nearest), the same
  // arithmetic as dequantize -> SoftmaxOperator<float> -> quantize. The
  // running maximum is taken over the dequantized values so the exponents
  // are always <= 0 (it used to be the raw int8 maximum, which underflowed
  // every exponent to 0 whenever the scale was small and returned 127 for
  // every class).
  const float beta_f = static_cast<float>(beta);
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }
  const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
  const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
  const float oscale = out->get_quantization_params().get_scale_for_channel(0);
  const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);

  for (int i = 0; i < out_side_numelems; i++) {
    // Linear indexing: the softmax runs over the last axis of any rank (the
    // 2D in(i, j) form indexed rank 3 and 4 tensors out of bounds).
    const uint32_t row = static_cast<uint32_t>(i) * depth;
    float max = std::numeric_limits<float>::lowest();
    for(int j = 0; j < depth; j++){
      const int32_t in32 = static_cast<int8_t>(in(row + j));
      max = std::max(max, in_scale * static_cast<float>(in32 - in_zp));
    }

    float mSum = 0;
    for(int j = 0; j < depth; j++){
      const int32_t in32 = static_cast<int8_t>(in(row + j));
      const float in_f = in_scale * static_cast<float>(in32 - in_zp);
      mSum += exp((in_f - max) * beta_f);
    }
    // Recomputes the exponentials instead of buffering them: no scratch RAM.
    // The input is re-read after outputs are written, so this kernel must
    // not run in place.
    for(int j = 0; j < depth; j++){
      const int32_t in32 = static_cast<int8_t>(in(row + j));
      const float in_f = in_scale * static_cast<float>(in32 - in_zp);
      const float out_val = exp((in_f - max) * beta_f) / mSum;
      const int32_t otmp = static_cast<int32_t>(std::round(out_val / oscale)) + ozp;
      const int8_t out8 = (otmp < -128) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);
      out(row + j)  = out8;
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
    Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
  softmax_k<T>(outT, inT, beta);
}

// Symmetric Quantized reference
template <>
inline void SoftmaxOperator<int8_t>::compute() {
  const Tensor& inT = inputs[in].tensor();
  Tensor& outT = outputs[out].tensor();
  // TODO Check sizes here and throw mismatch
  uint32_t in_size = inT->get_shape().get_linear_size();
  uint32_t out_size = outT->get_shape().get_linear_size();
  if (in_size != out_size)
    Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
  sq_softmax_k(outT, inT, beta);
}

} 
}  // namespace pTensor

#endif
