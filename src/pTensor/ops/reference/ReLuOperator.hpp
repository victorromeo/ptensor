#ifndef PTENSOR_ACTIVATIONS_OPS_RELU_H
#define PTENSOR_ACTIVATIONS_OPS_RELU_H

#include "activationOpBase.hpp"

namespace pTensor {

template <typename T>
class inplace_relu_k_impl {
  public:
    inplace_relu_k_impl() {}
    void operator()(Tensor& t) const;
};

template <typename T>
void inplace_relu_k_impl<T>::operator()(Tensor& t) const {
  T tmp;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    tmp = t(i);
    if (tmp < 0) {
      t(i) = static_cast<T>(0);
    }
  }
}

template <typename T>
class relu_k_impl{
  public:
    void operator()(Tensor& out, const Tensor& in) const;
};

template <>
void relu_k_impl<float>::operator()(Tensor& out, const Tensor& in) const;

// For all quantized forms
template <typename T>
void relu_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  static_assert(std::is_integral<T>::value, "Quantized ReLU expects integral type");
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  float tmp;
  uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    const int32_t iv8 = static_cast<T>(in(i));
    const float scale = in->get_quantization_params().get_scale_for_channel(0);
    const int32_t zp = in->get_quantization_params().get_zeroP_for_channel(0);
    tmp = (iv8 - zp)*scale;
    if (tmp < 0) {
      tmp = 0;
    }
    const float oscale = out->get_quantization_params().get_scale_for_channel(0);
    const int32_t ozp = out->get_quantization_params().get_zeroP_for_channel(0);
    const int32_t otmp = static_cast<int32_t>(tmp/oscale) + ozp;
    const T outT= (otmp <= min ) ? min : (otmp > max) ? max : static_cast<T>(otmp);
    out(i) = outT;
  }
}

template <typename T>
class inplace_relu6_k_impl {
  public:
    void operator()(Tensor& t) const;

};

template <typename T>
void inplace_relu6_k_impl<T>::operator()(Tensor& t) const {
  T tmp;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    tmp = t(i);
    if (tmp < 0) {
      t(i) = static_cast<T>(0);
    }
    if (tmp > 6) {
      t(i) = static_cast<T>(6);
    }
  }
}

template <typename T>
class relu6_k_impl {
  public:
    void operator()(Tensor& out, const Tensor& in) const;
};

template <typename T>
void relu6_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  T tmp;
  uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    tmp = in(i);
    if (tmp < 0) {
      tmp = static_cast<T>(0);
    }
    if (tmp > 6) {
      tmp = static_cast<T>(6);
    }
    out(i) = tmp;
  }
}

template <typename T>
using inplace_relu_k = inplace_relu_k_impl<T>;
template <typename T>
using relu_k = relu_k_impl<T>;
template <typename T>
using inplace_relu6_k = inplace_relu6_k_impl<T>();
template <typename T>
using relu6_k = relu6_k_impl<T>();

namespace ReferenceOperators {

template <typename T>
class InPlaceReLU : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { inplace_relu_k<T>()(inputs[x].tensor()); }
};

template <typename T>
class InPlaceReLU6 : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { inplace_relu6_k<T>()(inputs[x].tensor()); }
};

template <typename T>
class ReLUOperator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    relu_k<T>()(outT, inT);
  }
};

template <typename T>
class ReLU6Operator : public OperatorInterface<1, 1> {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[in].tensor();
    Tensor& outT = outputs[out].tensor();
    // TODO Check sizes here and throw mismatch
    uint32_t in_size = inT->get_shape().get_linear_size();
    uint32_t out_size = outT->get_shape().get_linear_size();
    if (in_size != out_size)
      Context::get_default_context()->throwError(
          new OperatorIOSizeMismatchError);
    relu6_k<T>(outT, inT);
  }
};

}

}  // namespace pTensor

#endif
