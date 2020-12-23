#ifndef PTENSOR_ACTIVATIONS_OPS_SIGMOID_H
#define PTENSOR_ACTIVATIONS_OPS_SIGMOID_H

#include <type_traits>

#include "activationOpBase.hpp"

namespace pTensor {

template <typename T>
class inplace_sigmoid_k {
  public:
    void operator()(Tensor& t) const;
};

template <typename T>
void inplace_sigmoid_k<T>::operator()(Tensor& t) const {
  const T one = 1;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(t(i))));
    t(i) = tmp;
  }
}

template <typename T>
class sigmoid_k_impl {
  public:
    void operator()(Tensor& out, const Tensor& in) const;

};

template <typename T>
void sigmoid_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  const T one = 1;
  uint32_t t_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(in(i))));
    out(i) = tmp;
  }
}

template <>
void sigmoid_k_impl<int8_t>::operator()(Tensor& out, const Tensor& in) const {
  const float one = 1;
  uint32_t t_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const int32_t in32 =  static_cast<int8_t>(in(i));
    const float in_scale = in->get_quantization_params().get_scale_for_channel(0);
    const int32_t in_zp = in->get_quantization_params().get_zeroP_for_channel(0);
    const float in_f = (in32 - in_zp)*in_scale;
    const float out_val = one / (one + exp( -in_f ));
    const float oscale = out->get_quantization_params().get_scale_for_channel(0);
    const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);
    const int32_t otmp = static_cast<int32_t>(out_val/oscale) + ozp;
    const int8_t out8 = (otmp < -127 ) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);  
    out(i) = out8;
  }
}

// Set defaults
template <typename T>
using sigmoid_k = sigmoid_k_impl<T>;

}  // namespace pTensor
#endif
