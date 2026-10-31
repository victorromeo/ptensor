#include "fully_connected_kernel.hpp"

#include "symmetric_quantization_utils.hpp"
#include "pTensor/core/fast_access.hpp"

namespace pTensor {
namespace TFLM {

namespace {

// output[b, o] = clamp(requant_o(sum_d (w[d, o] + w_off) * (x[b, d] + x_off)
//                                + bias[o]) + out_off)
// with filter laid out [in, out]. `mult`/`shift` hold either one entry
// (per-tensor) or one per output channel; `stride` is 0 or 1 accordingly.
void quantized_fc_impl(Tensor& output, const Tensor& input,
                       const Tensor& filter,
                       const std::function<int32_t(int32_t)>& bias,
                       const int32_t* mult, const int32_t* shift, int stride,
                       int32_t output_activation_min,
                       int32_t output_activation_max) {
  const int32_t input_offset =
      -input->get_quantization_params().get_zeroP_for_channel(0);
  const int32_t filter_offset =
      -filter->get_quantization_params().get_zeroP_for_channel(0);
  const int32_t output_offset =
      output->get_quantization_params().get_zeroP_for_channel(0);

  const TensorShape& filter_shape = filter->get_shape();
  TensorShape& output_shape = output->get_shape();

  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  const int accum_depth = filter_shape[0];
  if (filter_shape[1] != output_depth ||
      input->get_shape()[1] != accum_depth) {
    Context::get_default_context()->throwError(InvalidMatrixMultIndicesError());
    return;
  }
  const int8_t* px = fast_read<int8_t>(input);
  const int8_t* pw = fast_read<int8_t>(filter);
  int8_t* py = fast_write<int8_t>(output);
  const bool fast = px && pw && py;
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      if (fast) {
        // Dense arrays: same integer sum without per-element virtual reads.
        const int8_t* xrow = px + (size_t)b * accum_depth;
        const int8_t* wcol = pw + out_c;
        for (int d = 0; d < accum_depth; ++d) {
          acc += (static_cast<int32_t>(wcol[(size_t)d * output_depth]) + filter_offset) *
                 (static_cast<int32_t>(xrow[d]) + input_offset);
        }
      } else {
        for (int d = 0; d < accum_depth; ++d) {
          int32_t input_val = static_cast<int8_t>(input(b, d));
          int32_t filter_val = static_cast<int8_t>(filter(d, out_c));
          acc += (filter_val + filter_offset) * (input_val + input_offset);
        }
      }
      acc += static_cast<int32_t>(bias(out_c));
      acc = MultiplyByQuantizedMultiplier(acc, mult[out_c * stride],
                                          shift[out_c * stride]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      if (fast)
        py[(size_t)b * output_depth + out_c] = static_cast<int8_t>(acc);
      else
        output(b, out_c) = static_cast<int8_t>(acc);
    }
  }
}

}  // namespace

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, std::function<int32_t(int32_t)> bias,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max) {
  // Callers pass the right shift (negated TFLite exponent).
  const int32_t shift = -output_shift;
  quantized_fc_impl(output, input, filter, bias, &output_multiplier, &shift, 0,
                    output_activation_min, output_activation_max);
}

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, std::function<int32_t(int32_t)> bias,
                                  const int32_t* output_multiplier,
                                  const int32_t* output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max) {
  quantized_fc_impl(output, input, filter, bias, output_multiplier,
                    output_shift, 1, output_activation_min,
                    output_activation_max);
}

}  // namespace TFLM
}  // namespace pTensor
