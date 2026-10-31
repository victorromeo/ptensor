#ifndef PTENSOR_ACTIVATIONS_OPS_TANH_H
#define PTENSOR_ACTIVATIONS_OPS_TANH_H

#include <cmath>
#include <limits>
#include <type_traits>

#include "activationOpBase.hpp"
#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"

namespace pTensor {
namespace ReferenceOperators {

namespace tanh_detail {

// Floating point element types are used as-is; integer element types are
// treated as affine-quantized values (real = (q - zp) * scale).
template <typename T, bool IsFloat = std::is_floating_point<T>::value>
struct codec;

template <typename T>
struct codec<T, true> {
  static float decode(const Tensor& t, uint32_t i) {
    return static_cast<float>(static_cast<T>(t(i)));
  }
  static void encode(Tensor& t, uint32_t i, float v) { t(i) = static_cast<T>(v); }
};

template <typename T>
struct codec<T, false> {
  static float decode(const Tensor& t, uint32_t i) {
    const QuantizationParams& qp = t->get_quantization_params();
    const int32_t q = static_cast<T>(t(i));
    return (q - qp.get_zeroP_for_channel(0)) * qp.get_scale_for_channel(0);
  }
  static void encode(Tensor& t, uint32_t i, float v) {
    const QuantizationParams& qp = t->get_quantization_params();
    const int32_t q = static_cast<int32_t>(
                          std::lround(v / qp.get_scale_for_channel(0))) +
                      qp.get_zeroP_for_channel(0);
    const int32_t lo = std::numeric_limits<T>::min();
    const int32_t hi = std::numeric_limits<T>::max();
    t(i) = static_cast<T>(q < lo ? lo : (q > hi ? hi : q));
  }
};

}  // namespace tanh_detail

/**
 * Hyperbolic tangent activation.
 *
 * TanhOperator<float> computes tanh directly. For integer types, the input and
 * output tensors must carry per-tensor quantization parameters.
 */
template <typename Tin, typename Tout = Tin>
class TanhOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { act_in };
  enum names_out : uint8_t { act_out };

 protected:
  virtual void compute() {
    const Tensor& inT = inputs[act_in].tensor();
    Tensor& outT = outputs[act_out].tensor();
    const uint32_t n = inT->get_shape().get_linear_size();
    if (n != outT->get_shape().get_linear_size()) {
      Context::get_default_context()->throwError(OperatorIOSizeMismatchError());
      return;
    }
    for (uint32_t i = 0; i < n; i++) {
      const float x = tanh_detail::codec<Tin>::decode(inT, i);
      tanh_detail::codec<Tout>::encode(outT, i, std::tanh(x));
    }
  }
};

}  // namespace ReferenceOperators
}  // namespace pTensor

#endif
