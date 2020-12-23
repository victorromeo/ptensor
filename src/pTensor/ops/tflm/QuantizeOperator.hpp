#ifndef PTENSOR_QUANTIZE_OPS_H
#define PTENSOR_QUANTIZE_OPS_H
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {
namespace TflmSymQuantOps {

template <typename Tout, typename Tin>
void affine_quantize_kernel(Tensor& output, const Tensor& input) {
  // op:
  // https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/micro/kernels/quantize.cc
  // kernel:
  // https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/kernels/internal/reference/quantize.h
  const QuantizationParams& quant_params = output->get_quantization_params();
  if (output->num_elems() == 0) {
    output->resize(input->get_shape());
  }
  if (input->num_elems() != output->num_elems()) {
    pTensor_printf(
        "number of elements of output tensor mismatch with the input for "
        "quantization\n");
    Context::get_default_context()->throwError(new InvalidTensorOutputError);
    return;
  }
  const int32_t zp = quant_params.get_zeroP_for_channel(0);
  const float scale = quant_params.get_scale_for_channel(0);
  const int32_t minVal = static_cast<int32_t>(std::numeric_limits<Tout>::min());
  const int32_t maxVal = static_cast<int32_t>(std::numeric_limits<Tout>::max());
  for (uint32_t i = 0; i < input->num_elems(); i++) {
    const Tin inVal = input(i);
    const float inVal_f = static_cast<float>(inVal);
    int32_t unclamped = static_cast<int32_t>(std::round(inVal_f / scale)) + zp;
    int32_t clamped = std::min(std::max(unclamped, minVal), maxVal);
    output(i) = static_cast<Tout>(clamped);
  }
}

// TODO @mbartling  Add template specializations for invalid type combos to
// sanity check
template <typename Tout, typename Tin>
class QuantizeOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };

 protected:
  void compute() {
    affine_quantize_kernel<Tout, Tin>(outputs[output].tensor(),
                                      inputs[input].tensor());
  }
};

}  // namespace TFLM
}  // namespace pTensor

#endif
