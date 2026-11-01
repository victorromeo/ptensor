#ifndef S_QUANTIZED_CONVOLUTION_KERNELS_HPP
#define S_QUANTIZED_CONVOLUTION_KERNELS_HPP
#include "context.hpp"
#include "symmetric_quantization_utils.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "pTensor_util.hpp"
#include "convolution_helper.hpp"
#include "ConvolutionOperator.hpp"
#include "pTensor/core/fast_access.hpp"

namespace pTensor {
namespace TflmSymQuantOps {
// Basically #defines for shape dimensions
using namespace pTensor::ReferenceOperators::Conv2dConstants;

template <typename T>
void squantize_convolution_kernel(Tensor& out, const Tensor& in, 
                                const Tensor& filter,
                                const Tensor& bias, bool have_bias,
                                const Padding padding,
                                const uint16_t (&strides)[4],
                                const int32_t* output_multiplier,
                                const int32_t* output_shift,
                                const int32_t input_offset,
                                const int32_t output_offset,
                                const int32_t output_activation_min,
                                const int32_t output_activation_max,
                                const int dilation_rows = 1,
                                const int dilation_cols = 1) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& f_shape  = filter->get_shape();

  const int16_t input_depth   = in_shape[input_channel_dim];
  const int16_t input_rows    = in_shape[input_height_dim];
  const int16_t input_cols    = in_shape[input_witdh_dim];
  const int16_t input_batches = in_shape[input_batch_dim];
  const int16_t out_depth     = f_shape[filter_out_channels_dim]; 
  const int16_t filter_rows   = f_shape[filter_height_dim];
  const int16_t filter_cols   = f_shape[filter_width_dim];
  const int16_t filter_count  = f_shape[filter_out_channels_dim];

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[output_height_dim];
  int16_t out_cols = out->get_shape()[output_width_dim];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  // TensorFlow padding (it used to be negative, and so shift the window,
  // when the stride was larger than the kernel).
  // Dilation was accepted by Conv2dOperator but ignored here.
  const int filter_left_offset = padding_before(
      input_cols, out_cols, filter_cols, stride_cols, dilation_cols, padding);
  const int filter_top_offset = padding_before(
      input_rows, out_rows, filter_rows, stride_rows, dilation_rows, padding);

  // Fast path: every tensor is a dense array, so walk raw pointers. Integer
  // accumulation, so the result is identical to the reference loop below.
  {
    const T* px = fast_read<T>(in);
    const T* pf = fast_read<T>(filter);
    const int32_t* pb = have_bias ? fast_read<int32_t>(bias) : nullptr;
    T* py = fast_write<T>(out);
    if (px && pf && py && (!have_bias || pb)) {
      for (int batch = 0; batch < input_batches; ++batch) {
        const T* pxb = px + (size_t)batch * input_rows * input_cols * input_depth;
        for (int out_y = 0; out_y < out_rows; ++out_y) {
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          for (int out_x = 0; out_x < out_cols; ++out_x) {
            const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
            for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
                const int in_y = in_y_origin + dilation_rows * filter_y;
                if (in_y < 0 || in_y >= input_rows) continue;
                for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
                  const int in_x = in_x_origin + dilation_cols * filter_x;
                  if (in_x < 0 || in_x >= input_cols) continue;
                  const T* xv = pxb + ((size_t)in_y * input_cols + in_x) * input_depth;
                  const T* fv = pf + (((size_t)out_channel * filter_rows + filter_y) *
                                          filter_cols + filter_x) * input_depth;
                  for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    acc += static_cast<int32_t>(fv[in_channel]) *
                           (static_cast<int32_t>(xv[in_channel]) + input_offset);
                  }
                }
              }
              if (have_bias) acc += pb[out_channel];
              acc = TFLM::MultiplyByQuantizedMultiplier(
                  acc, output_multiplier[out_channel], output_shift[out_channel]);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              *py++ = static_cast<T>(acc);
            }
          }
        }
      }
      return;
    }
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Each filter kernel produces one output channel.
        for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          // T output_val = 0;
          //filter.reset();
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_cols * filter_x;
                const int in_y = in_y_origin + dilation_rows * filter_y;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  int32_t input_value = static_cast<T>(in(batch, in_y, in_x, in_channel));
                  int32_t filter_value = static_cast<T>(filter(out_channel, filter_y, filter_x, in_channel));
                  acc += filter_value * (input_value + input_offset);

                } // Inside image
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          if(have_bias){
            acc += static_cast<int32_t>(bias(out_channel));
          }
          acc = TFLM::MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel],
              output_shift[out_channel]);
          acc += output_offset;
          // clamp
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          out(batch, out_y, out_x, out_channel) = static_cast<T>(acc);
        } //For out_channel
      } // For out cols
    } // For out rows
  } // For input batches
} //Kernel

} // TflmSymQuantOps
} // pTensor

#endif
