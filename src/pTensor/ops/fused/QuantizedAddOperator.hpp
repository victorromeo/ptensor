#ifndef PTENSOR_QUANTIZED_ADD_OPS_H
#define PTENSOR_QUANTIZED_ADD_OPS_H
/**
 * int8 add with the activation and the requantisation to the output scale
 * fused in, bit-exact with TFLite's ADD (reference integer kernel, int8).
 *
 * Both inputs are rescaled to a common fixed-point scale (left shift 20, as
 * TFLite), added, requantised once to the output scale and clamped to the
 * activation range. A residual "add then ReLU" therefore costs one pass and
 * one rounding instead of two.
 *
 * Broadcasting follows NumPy over up to 4 dimensions. Quantisation must be
 * per tensor (as TFLite requires for ADD).
 */
#include <limits>

#include "pTensor/core/context.hpp"
#include "pTensor/core/fast_access.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/ops/fast/fused_kernels.hpp"
#include "pTensor/ops/reference/ElementwiseOperators.hpp"
#include "pTensor/ops/tflm/symmetric_quantization_utils.hpp"
#include "pTensor/ops/tflm/tflm_defs.hpp"
#include "pTensor/util/fixed_point.hpp"

namespace pTensor {
namespace TflmSymQuantOps {

namespace qadd_detail {
inline int32_t mbqm(int32_t x, int32_t multiplier, int32_t shift) {
  return fixed_point::multiply_by_quantized_multiplier(x, multiplier, shift);
}
}  // namespace qadd_detail

/** Fixed-point parameters of an int8 add; computed once per eval. */
struct QuantizedAddParams {
  int32_t left_shift;
  int32_t input1_offset, input1_multiplier, input1_shift;
  int32_t input2_offset, input2_multiplier, input2_shift;
  int32_t output_offset, output_multiplier, output_shift;
  int32_t act_min, act_max;

  inline int8_t operator()(int8_t x1, int8_t x2) const {
    const int32_t shifted1 = (input1_offset + x1) * (1 << left_shift);
    const int32_t shifted2 = (input2_offset + x2) * (1 << left_shift);
    const int32_t scaled1 =
        qadd_detail::mbqm(shifted1, input1_multiplier, input1_shift);
    const int32_t scaled2 =
        qadd_detail::mbqm(shifted2, input2_multiplier, input2_shift);
    int32_t r = qadd_detail::mbqm(scaled1 + scaled2, output_multiplier,
                                  output_shift) +
                output_offset;
    r = r < act_min ? act_min : (r > act_max ? act_max : r);
    return static_cast<int8_t>(r);
  }

  // Every shift within what the fixed-point helpers support.
  bool valid() const {
    const int32_t s[3] = {input1_shift, input2_shift, output_shift};
    for (int i = 0; i < 3; ++i)
      if (s[i] < -31 || s[i] > 30) return false;
    return true;
  }
};

/** TFLite's Prepare() for int8 ADD. */
inline QuantizedAddParams make_quantized_add_params(
    float s1, int32_t zp1, float s2, int32_t zp2, Tensor& out,
    TFLM::TfLiteFusedActivation activation) {
  QuantizedAddParams p;
  p.left_shift = 20;
  const float so = out->get_quantization_params().get_scale_for_channel(0);
  const double twice_max = 2.0 * static_cast<double>(s1 > s2 ? s1 : s2);
  int shift;
  TFLM::QuantizeMultiplier(static_cast<double>(s1) / twice_max,
                           &p.input1_multiplier, &shift);
  p.input1_shift = shift;
  TFLM::QuantizeMultiplier(static_cast<double>(s2) / twice_max,
                           &p.input2_multiplier, &shift);
  p.input2_shift = shift;
  TFLM::QuantizeMultiplier(
      twice_max / ((1 << p.left_shift) * static_cast<double>(so)),
      &p.output_multiplier, &shift);
  p.output_shift = shift;
  p.input1_offset = -zp1;
  p.input2_offset = -zp2;
  p.output_offset = out->get_quantization_params().get_zeroP_for_channel(0);
  TFLM::CalculateActivationRangeQuantized(activation, out, &p.act_min,
                                          &p.act_max);
  return p;
}

template <typename T>
class QuantizedAddOperator : public OperatorInterface<2, 1> {};

template <>
class QuantizedAddOperator<int8_t> : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };

  explicit QuantizedAddOperator(
      TFLM::TfLiteFusedActivation activation = TFLM::kTfLiteActNone)
      : _activation(activation) {}

 protected:
  virtual void compute() {
    const Tensor& ta = inputs[a].tensor();
    const Tensor& tb = inputs[b].tensor();
    Tensor& tc = outputs[c].tensor();
    if (ta->get_type() != i8 || tb->get_type() != i8 || tc->get_type() != i8) {
      Context::get_default_context()->throwError(InvalidTensorDataTypeError());
      return;
    }
    const QuantizationParams& qa = ta->get_quantization_params();
    const QuantizationParams& qb = tb->get_quantization_params();
    if (qa.num_channels() > 1 || qb.num_channels() > 1 ||
        tc->get_quantization_params().num_channels() > 1) {
      Context::get_default_context()->throwError(
          InvalidQuantizationSchemeError());
      return;
    }
    uint32_t od[4];
    elementwise::right_align(tc->get_shape(), od);
    bool ok = true;
    const elementwise::BroadcastView va =
        elementwise::make_view(ta->get_shape(), od, ok);
    const elementwise::BroadcastView vb =
        elementwise::make_view(tb->get_shape(), od, ok);
    if (!ok) {
      Context::get_default_context()->throwError(BroadcastShapeMismatchError());
      return;
    }
    const QuantizedAddParams p = make_quantized_add_params(
        qa.get_scale_for_channel(0), qa.get_zeroP_for_channel(0),
        qb.get_scale_for_channel(0), qb.get_zeroP_for_channel(0), tc,
        _activation);
    if (!p.valid()) {
      Context::get_default_context()->throwError(
          SymmetricQuantizationFixedPointRangeError());
      return;
    }

    const int8_t* pa = fast_read<int8_t>(ta);
    const int8_t* pb = fast_read<int8_t>(tb);
    int8_t* pc = fast_write<int8_t>(tc);
    if (pa && pb && pc) {
      fast::broadcast_zip(pa, va.strides, pb, vb.strides, pc, od, p);
      return;
    }
    const fused::AnyIn<int8_t> xa(pa, &ta);
    const fused::AnyIn<int8_t> xb(pb, &tb);
    const fused::AnyOut<int8_t> xc(pc, &tc);
    uint32_t oi = 0;
    for (uint32_t i0 = 0; i0 < od[0]; ++i0)
      for (uint32_t i1 = 0; i1 < od[1]; ++i1)
        for (uint32_t i2 = 0; i2 < od[2]; ++i2) {
          const uint32_t base_a =
              i0 * va.strides[0] + i1 * va.strides[1] + i2 * va.strides[2];
          const uint32_t base_b =
              i0 * vb.strides[0] + i1 * vb.strides[1] + i2 * vb.strides[2];
          for (uint32_t i3 = 0; i3 < od[3]; ++i3, ++oi)
            xc.set(oi, p(xa[base_a + i3 * va.strides[3]],
                         xb[base_b + i3 * vb.strides[3]]));
        }
  }

 private:
  TFLM::TfLiteFusedActivation _activation;
};

}  // namespace TflmSymQuantOps
}  // namespace pTensor
#endif
