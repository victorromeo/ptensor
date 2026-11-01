#ifndef PTENSOR_ACTIVATIONS_OPS_RELU_H
#define PTENSOR_ACTIVATIONS_OPS_RELU_H

#include <algorithm>
#include <cmath>
#include <limits>

#include "activationOpBase.hpp"
#include "pTensor/core/context.hpp"
#include "pTensor/util/fixed_point.hpp"

namespace pTensor {

// Input / output scales of a quantized ReLU cannot be expressed as a
// fixed-point multiplier (zero, negative or out of range).
DECLARE_ERROR(QuantizedActivationScaleError);

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

// Float ReLU (was declared but never defined, so ReLUOperator<float> failed
// to link).
template <>
inline void relu_k_impl<float>::operator()(Tensor& out, const Tensor& in) const {
  const uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    const float v = static_cast<float>(in(i));
    out(i) = v < 0.0f ? 0.0f : v;
  }
}

namespace relu_detail {

/**
 * Quantized ReLU / ReLU6 with TFLite's integer arithmetic (activations.cc,
 * ReluQuantized): rescale from the input to the output quantization with a
 * fixed-point multiplier, then clamp to the activation range, so outputs are
 * bit-exact with TFLite. Per-tensor quantization on both tensors.
 */
template <typename T>
void quantized_relu(Tensor& out, const Tensor& in, bool relu6) {
  static_assert(std::is_integral<T>::value, "Quantized ReLU expects integral type");
  const QuantizationParams& qi = in->get_quantization_params();
  const QuantizationParams& qo = out->get_quantization_params();
  const float si = qi.get_scale_for_channel(0);
  const float so = qo.get_scale_for_channel(0);
  const int32_t zi = qi.get_zeroP_for_channel(0);
  const int32_t zo = qo.get_zeroP_for_channel(0);
  int32_t mult, shift;
  if (!(si > 0) || !(so > 0) ||
      !fixed_point::quantize_multiplier(static_cast<double>(si) / so, &mult, &shift)) {
    Context::get_default_context()->throwError(QuantizedActivationScaleError());
    return;
  }
  const int32_t qmin = std::numeric_limits<T>::lowest();
  const int32_t qmax = std::numeric_limits<T>::max();
  const int32_t act_min = std::max(qmin, zo + static_cast<int32_t>(std::round(0.0f / so)));
  const int32_t act_max =
      relu6 ? std::min(qmax, zo + static_cast<int32_t>(std::round(6.0f / so))) : qmax;
  const uint32_t n = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < n; i++) {
    const int32_t v = static_cast<T>(in(i));
    int32_t r = zo + fixed_point::multiply_by_quantized_multiplier(v - zi, mult, shift);
    r = r < act_min ? act_min : (r > act_max ? act_max : r);
    out(i) = static_cast<T>(r);
  }
}

}  // namespace relu_detail

// For all quantized forms
template <typename T>
void relu_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  relu_detail::quantized_relu<T>(out, in, false);
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

// Quantized forms: TFLite-exact, as ReLU above.
template <typename T>
void relu6_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  relu_detail::quantized_relu<T>(out, in, true);
}

template <>
inline void relu6_k_impl<float>::operator()(Tensor& out, const Tensor& in) const {
  float tmp;
  uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    tmp = in(i);
    if (tmp < 0) {
      tmp = 0.0f;
    }
    if (tmp > 6) {
      tmp = 6.0f;
    }
    out(i) = tmp;
  }
}

template <typename T>
using inplace_relu_k = inplace_relu_k_impl<T>;
template <typename T>
using relu_k = relu_k_impl<T>;
template <typename T>
using inplace_relu6_k = inplace_relu6_k_impl<T>;
template <typename T>
using relu6_k = relu6_k_impl<T>;

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
      Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
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
      Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
    relu6_k<T>()(outT, inT);
  }
};

}

}  // namespace pTensor

#endif
