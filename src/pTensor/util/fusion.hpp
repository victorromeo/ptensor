#ifndef PTENSOR_UTIL_FUSION_H
#define PTENSOR_UTIL_FUSION_H
/**
 * Batch-norm folding for the fused operators (ops/fused/FusedOperators.hpp).
 *
 * An inference-time batch norm after a conv / depthwise conv / dense layer is
 * a per-channel affine map, so
 *
 *   bn(layer(x) + bias) = layer(x) * s + ((bias - mean) * s + beta)
 *   with s = gamma / sqrt(variance + epsilon)
 *
 * fold_batch_norm() computes s and the new bias. Then either
 *
 *   - pass s as the fused operator's `scale` input and the new bias as its
 *     `bias` (weights untouched: works for weights in flash / packed), or
 *   - multiply s into the weights with fold_*_weights() and pass only the
 *     new bias (costs nothing at run time; needs writable float weights,
 *     e.g. on the host before export, or a RAM copy at start-up).
 *
 * Plain functions over float arrays: no tensors, no allocation. Python has
 * the same in ptensor.fusion.
 */
#include <cmath>
#include <stdint.h>

namespace pTensor {
namespace fusion {

/**
 * scale[c] = gamma[c] / sqrt(variance[c] + epsilon)
 * bias_out[c] = (bias_in[c] - mean[c]) * scale[c] + beta[c]
 * bias_in may be null (layer without bias); bias_out may alias bias_in.
 * gamma / beta may be null (Keras center=False / scale=False): 1 and 0.
 */
inline void fold_batch_norm(const float* mean, const float* variance,
                            const float* gamma, const float* beta,
                            float epsilon, const float* bias_in,
                            uint32_t channels, float* scale, float* bias_out) {
  for (uint32_t c = 0; c < channels; ++c) {
    const float g = gamma ? gamma[c] : 1.0f;
    const float s = g / std::sqrt(variance[c] + epsilon);
    const float b = bias_in ? bias_in[c] : 0.0f;
    scale[c] = s;
    bias_out[c] = (b - mean[c]) * s + (beta ? beta[c] : 0.0f);
  }
}

/** Conv2d OHWI filter [out_c, kh, kw, in_c]: row oc *= scale[oc]. */
inline void fold_conv2d_weights(float* filter, uint32_t out_c,
                                uint32_t per_channel, const float* scale) {
  for (uint32_t oc = 0; oc < out_c; ++oc) {
    float* row = filter + oc * per_channel;
    for (uint32_t i = 0; i < per_channel; ++i) row[i] *= scale[oc];
  }
}

/** Depthwise filter [1, kh, kw, C]: element i *= scale[i % C]. */
inline void fold_depthwise_weights(float* filter, uint32_t taps,
                                   uint32_t channels, const float* scale) {
  for (uint32_t t = 0; t < taps; ++t) {
    float* row = filter + t * channels;
    for (uint32_t c = 0; c < channels; ++c) row[c] *= scale[c];
  }
}

/** Dense weights [in, out]: column o *= scale[o]. */
inline void fold_dense_weights(float* weights, uint32_t in_features,
                               uint32_t out_features, const float* scale) {
  fold_depthwise_weights(weights, in_features, out_features, scale);
}

}  // namespace fusion
}  // namespace pTensor
#endif
