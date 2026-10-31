#ifndef PTENSOR_S_QUANTIZED_FC_OPS_H
#define PTENSOR_S_QUANTIZED_FC_OPS_H

#include "MatrixMultiplyOperator.hpp"
#include "context.hpp"
#include "fully_connected_kernel.hpp"
#include "operatorBase.hpp"
#include "symmetric_quantization_utils.hpp"

namespace pTensor {
namespace TflmSymQuantOps {

template <typename T>
class QuantizedMatrixMultiplyOperator : public OperatorInterface<3, 1> {};

template <>
class QuantizedMatrixMultiplyOperator<int8_t> : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { input, filter, bias };
  enum names_out : uint8_t { output };

  QuantizedMatrixMultiplyOperator(TFLM::TfLiteFusedActivation activation)
      : _activation(activation) {}

 private:
  TFLM::TfLiteFusedActivation _activation;

 protected:
  virtual void compute() {
    bool have_bias = inputs.has(bias);
    // Decide on c shape
    TensorShape& a_shape = inputs[input].tensor()->get_shape();
    TensorShape& b_shape = inputs[filter].tensor()->get_shape();
    TensorShape& c_shape = outputs[output].tensor()->get_shape();
    if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
        c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
        a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
      pTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
      Context::get_default_context()->throwError(pTensor::InvalidMatrixMultIndicesError());
      return;
    }
    if (inputs[input].tensor()->get_type() != i8 ||
        inputs[filter].tensor()->get_type() != i8 ||
        outputs[output].tensor()->get_type() != i8 ||
        (have_bias && inputs[bias].tensor()->get_type() != i32)) {
      Context::get_default_context()->throwError(InvalidTensorDataTypeError());
      return;
    }

    int32_t output_activation_min;
    int32_t output_activation_max;
    TFLM::CalculateActivationRangeQuantized(
        _activation, outputs[output].tensor(), &output_activation_min,
        &output_activation_max);

    const Tensor& b = inputs[bias].tensor();
    std::function<int32_t(int32_t)> bias_fn;
    if (have_bias) {
      bias_fn = [&b](int32_t i) { return static_cast<int32_t>(b(i)); };
    } else {
      bias_fn = [](int32_t) { return 0; };
    }

    const QuantizationParams& wq =
        inputs[filter].tensor()->get_quantization_params();
    if (wq.num_channels() > 1) {
      // Per-channel weights (TFLite's default for int8 dense layers since
      // TF 2.16): one requantization multiplier per output channel.
      const int channels = b_shape[1];
      if (wq.num_channels() != channels) {
        Context::get_default_context()->throwError(
            InvalidQuantizationSchemeError());
        return;
      }
      AllocatorInterface* ram =
          Context::get_default_context()->get_ram_data_allocator();
      int32_t* mult = reinterpret_cast<int32_t*>(
          ram->allocate(2 * sizeof(int32_t) * channels));
      int32_t* shift = mult + channels;
      const double in_scale = static_cast<double>(
          inputs[input].tensor()->get_quantization_params().get_scale_for_channel(0));
      const double out_scale = static_cast<double>(
          outputs[output].tensor()->get_quantization_params().get_scale_for_channel(0));
      for (int c = 0; c < channels; ++c) {
        int exponent;
        TFLM::QuantizeMultiplier(
            in_scale * static_cast<double>(wq.get_scale_for_channel(c)) /
                out_scale,
            &mult[c], &exponent);
        shift[c] = exponent;
      }
      TFLM::quantized_matrix_mult_kernel(
          outputs[output].tensor(), inputs[input].tensor(),
          inputs[filter].tensor(), bias_fn, mult, shift,
          output_activation_min, output_activation_max);
      ram->deallocate(mult);
      return;
    }

    // Per-tensor weights. Taken from TFLu to maintain operator compatibility:
    // the scaling factor from input to output (aka the 'real multiplier') is
    // represented as a fixed point multiplier plus a shift.
    int32_t output_multiplier;
    int exponent;
    double real_multiplier = 0.0;
    TFLM::GetQuantizedConvolutionMultipler(
        inputs[input].tensor(), inputs[filter].tensor(), inputs[bias].tensor(),
        outputs[output].tensor(), &real_multiplier);
    TFLM::QuantizeMultiplier(real_multiplier, &output_multiplier, &exponent);
    TFLM::quantized_matrix_mult_kernel(
        outputs[output].tensor(), inputs[input].tensor(),
        inputs[filter].tensor(), bias_fn, output_multiplier, -exponent,
        output_activation_min, output_activation_max);
  }
};

template <typename Tout>
using FullyConnectedOperator = QuantizedMatrixMultiplyOperator<Tout>;

}
}  // namespace pTensor
#endif
