#ifndef PTENSOR_C_API_H
#define PTENSOR_C_API_H
/**
 * Plain C interface to pTensor.
 *
 * Each call runs one operator on caller-owned, contiguous, row-major buffers
 * (images are NHWC, conv filters are OHWI). Nothing is retained between calls.
 * The interface is meant for host-side verification (it backs the Python
 * package) and for C projects; it is not thread safe.
 *
 * Every function returns PTENSOR_OK (0) on success. On failure it returns a
 * nonzero status, and ptensor_last_error() gives pTensor's error id.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define PTENSOR_C_EXPORT __declspec(dllexport)
#else
#define PTENSOR_C_EXPORT __attribute__((visibility("default")))
#endif

typedef enum {
  PTENSOR_OK = 0,
  PTENSOR_ERR_INVALID_ARGUMENT = 1,  // Bad shape, size or parameter
  PTENSOR_ERR_RUNTIME = 2,           // pTensor raised an error while running
} ptensor_status;

typedef enum {
  PTENSOR_PADDING_VALID = 1,
  PTENSOR_PADDING_SAME = 2,
} ptensor_padding;

typedef enum {
  PTENSOR_ACT_NONE = 0,
  PTENSOR_ACT_RELU = 1,
  PTENSOR_ACT_RELU6 = 3,
} ptensor_activation;

/* Must match pTensor::PackedFormat. */
typedef enum {
  PTENSOR_PACKED_UINT_BITS = 0,
  PTENSOR_PACKED_INT8 = 1,
  PTENSOR_PACKED_UINT8 = 2,
  PTENSOR_PACKED_INT16 = 3,
  PTENSOR_PACKED_UINT16 = 4,
  PTENSOR_PACKED_INT32 = 5,
  PTENSOR_PACKED_FLOAT16 = 6,
  PTENSOR_PACKED_FLOAT32 = 7,
  PTENSOR_PACKED_FLOAT64 = 8,
} ptensor_packed_format;

PTENSOR_C_EXPORT const char* ptensor_version(void);
/** pTensor error id raised by the most recent failing call (0 if none). */
PTENSOR_C_EXPORT uint32_t ptensor_last_error(void);

/** Output spatial size for a conv/pool window, or 0 if the arguments are bad. */
PTENSOR_C_EXPORT uint32_t ptensor_conv_output_size(uint32_t in_size,
                                                   uint32_t kernel,
                                                   uint32_t stride,
                                                   int padding);

/* ---- float32 operators ------------------------------------------------ */

/** out[m,n] = a[m,k] @ b[k,n] */
PTENSOR_C_EXPORT int ptensor_matmul_f32(const float* a, const float* b,
                                        float* out, uint32_t m, uint32_t k,
                                        uint32_t n);

/** out[batch,out] = act(in[batch,in] @ weights[in,out] + bias[out]); bias may be NULL */
PTENSOR_C_EXPORT int ptensor_fully_connected_f32(
    const float* in, const float* weights, const float* bias, float* out,
    uint32_t batch, uint32_t in_features, uint32_t out_features, int activation);

/** Elementwise out = a + b over `count` elements. */
PTENSOR_C_EXPORT int ptensor_add_f32(const float* a, const float* b,
                                     float* out, uint32_t count);

PTENSOR_C_EXPORT int ptensor_relu_f32(const float* in, float* out,
                                      uint32_t count);
PTENSOR_C_EXPORT int ptensor_relu6_f32(const float* in, float* out,
                                       uint32_t count);
PTENSOR_C_EXPORT int ptensor_sigmoid_f32(const float* in, float* out,
                                         uint32_t count);
PTENSOR_C_EXPORT int ptensor_tanh_f32(const float* in, float* out,
                                      uint32_t count);
/** Softmax over the last axis of an [rows, depth] matrix. */
PTENSOR_C_EXPORT int ptensor_softmax_f32(const float* in, float* out,
                                         uint32_t rows, uint32_t depth);

/**
 * 2D convolution. in: [n,h,w,c], filter: [out_c,kh,kw,c], bias: [out_c] or
 * NULL, out: [n,out_h,out_w,out_c] where out_h/out_w follow
 * ptensor_conv_output_size.
 */
PTENSOR_C_EXPORT int ptensor_conv2d_f32(const float* in, uint32_t n,
                                        uint32_t h, uint32_t w, uint32_t c,
                                        const float* filter, uint32_t out_c,
                                        uint32_t kh, uint32_t kw,
                                        const float* bias, uint32_t stride_h,
                                        uint32_t stride_w, int padding,
                                        float* out);

/** Max / average pooling. in: [n,h,w,c], out: [n,out_h,out_w,c]. */
PTENSOR_C_EXPORT int ptensor_maxpool2d_f32(const float* in, uint32_t n,
                                           uint32_t h, uint32_t w, uint32_t c,
                                           uint32_t kh, uint32_t kw,
                                           uint32_t stride_h, uint32_t stride_w,
                                           int padding, float* out);
PTENSOR_C_EXPORT int ptensor_avgpool2d_f32(const float* in, uint32_t n,
                                           uint32_t h, uint32_t w, uint32_t c,
                                           uint32_t kh, uint32_t kw,
                                           uint32_t stride_h, uint32_t stride_w,
                                           int padding, float* out);

/* ---- extended float operators ---------------------------------------- */

typedef enum {
  PTENSOR_BINARY_ADD = 0,
  PTENSOR_BINARY_SUB = 1,
  PTENSOR_BINARY_MUL = 2,
  PTENSOR_BINARY_DIV = 3,
  PTENSOR_BINARY_MAXIMUM = 4,
  PTENSOR_BINARY_MINIMUM = 5,
  PTENSOR_BINARY_SQUARED_DIFFERENCE = 6,
  PTENSOR_BINARY_POW = 7,
} ptensor_binary_op;

typedef enum {
  PTENSOR_UNARY_LEAKY_RELU = 0, /* p0 = alpha */
  PTENSOR_UNARY_ELU = 1,        /* p0 = alpha */
  PTENSOR_UNARY_SELU = 2,
  PTENSOR_UNARY_HARD_SIGMOID = 3,
  PTENSOR_UNARY_HARD_SWISH = 4,
  PTENSOR_UNARY_SWISH = 5,
  PTENSOR_UNARY_GELU = 6,
  PTENSOR_UNARY_SOFTPLUS = 7,
  PTENSOR_UNARY_SOFTSIGN = 8,
  PTENSOR_UNARY_ABS = 9,
  PTENSOR_UNARY_NEG = 10,
  PTENSOR_UNARY_SQUARE = 11,
  PTENSOR_UNARY_SQRT = 12,
  PTENSOR_UNARY_RSQRT = 13,
  PTENSOR_UNARY_EXP = 14,
  PTENSOR_UNARY_LOG = 15,
  PTENSOR_UNARY_FLOOR = 16,
  PTENSOR_UNARY_CEIL = 17,
  PTENSOR_UNARY_ROUND = 18,
  PTENSOR_UNARY_SIN = 19,
  PTENSOR_UNARY_COS = 20,
  PTENSOR_UNARY_CLIP = 21, /* p0 = min, p1 = max */
} ptensor_unary_op;

/**
 * Elementwise binary op with NumPy broadcasting (rank <= 4). `out` must hold
 * the broadcast shape of a and b; incompatible shapes return
 * PTENSOR_INVALID_ARGUMENT.
 */
PTENSOR_C_EXPORT int ptensor_binary_f32(int op, const float* a,
                                        const uint32_t* a_shape,
                                        uint32_t a_rank, const float* b,
                                        const uint32_t* b_shape,
                                        uint32_t b_rank, float* out);
PTENSOR_C_EXPORT int ptensor_unary_f32(int op, const float* in, float* out,
                                       uint32_t count, float p0, float p1);

/** out = (in - mean) / sqrt(var + eps) * gamma + beta over the last axis. */
PTENSOR_C_EXPORT int ptensor_batchnorm_f32(const float* in, uint32_t count,
                                           uint32_t channels, const float* mean,
                                           const float* variance,
                                           const float* gamma,
                                           const float* beta, float epsilon,
                                           float* out);

/** Constant-pad an NHWC tensor in H and W. */
PTENSOR_C_EXPORT int ptensor_pad2d_f32(const float* in, uint32_t n, uint32_t h,
                                       uint32_t w, uint32_t c, uint32_t top,
                                       uint32_t bottom, uint32_t left,
                                       uint32_t right, float value, float* out);

/** Concatenate a and b (same rank <= 4) along axis. */
PTENSOR_C_EXPORT int ptensor_concat_f32(const float* a, const uint32_t* a_shape,
                                        const float* b, const uint32_t* b_shape,
                                        uint32_t rank, int axis, float* out);

/** [n,h,w,c] -> [n,c] spatial mean. */
PTENSOR_C_EXPORT int ptensor_global_avgpool_f32(const float* in, uint32_t n,
                                                uint32_t h, uint32_t w,
                                                uint32_t c, float* out);

/**
 * Depthwise conv. in: [n,h,w,c], filter: [1,kh,kw,c*mult], bias: [c*mult] or
 * NULL, out: [n,out_h,out_w,c*mult] (out size from ptensor_conv_output_size
 * using the dilated kernel size (k-1)*d+1).
 */
PTENSOR_C_EXPORT int ptensor_depthwise_conv2d_f32(
    const float* in, uint32_t n, uint32_t h, uint32_t w, uint32_t c,
    const float* filter, uint32_t kh, uint32_t kw, uint32_t multiplier,
    const float* bias, uint32_t stride_h, uint32_t stride_w, int padding,
    uint32_t dilation_h, uint32_t dilation_w, float* out);

/** One LSTM step (Keras gate order i,f,c,o). W: [in,4H], U: [H,4H], b: [4H]. */
PTENSOR_C_EXPORT int ptensor_lstm_cell_f32(const float* x, const float* h,
                                           const float* c, const float* W,
                                           const float* U, const float* b,
                                           uint32_t batch, uint32_t in_features,
                                           uint32_t hidden, float* h_out,
                                           float* c_out);

/** One GRU step (Keras z,r,h; reset_after). W: [in,3H], U: [H,3H]. */
PTENSOR_C_EXPORT int ptensor_gru_cell_f32(const float* x, const float* h,
                                          const float* W, const float* U,
                                          const float* bx, const float* bh,
                                          uint32_t batch, uint32_t in_features,
                                          uint32_t hidden, float* h_out);

/* ---- int8 quantization ------------------------------------------------ */

/** q = clamp(round(x / scale) + zero_point) */
PTENSOR_C_EXPORT int ptensor_quantize_i8(const float* in, int8_t* out,
                                         uint32_t count, float scale,
                                         int32_t zero_point);
/** x = (q - zero_point) * scale */
PTENSOR_C_EXPORT int ptensor_dequantize_i8(const int8_t* in, float* out,
                                           uint32_t count, float scale,
                                           int32_t zero_point);

/**
 * TFLite-compatible int8 fully connected layer.
 * in: [batch,in] (in_scale, in_zp); weights: [in,out] per-tensor symmetric
 * (w_scale); bias: [out] int32 with scale in_scale*w_scale, or NULL;
 * out: [batch,out] (out_scale, out_zp).
 */
PTENSOR_C_EXPORT int ptensor_fully_connected_i8(
    const int8_t* in, float in_scale, int32_t in_zp, const int8_t* weights,
    float w_scale, const int32_t* bias, int8_t* out, float out_scale,
    int32_t out_zp, uint32_t batch, uint32_t in_features,
    uint32_t out_features, int activation);

/**
 * Like ptensor_fully_connected_i8, but with per-output-channel weight scales
 * (w_scales has out_features entries; TFLite's default for int8 dense layers).
 * bias scale for channel o is in_scale*w_scales[o].
 */
PTENSOR_C_EXPORT int ptensor_fully_connected_i8_pc(
    const int8_t* in, float in_scale, int32_t in_zp, const int8_t* weights,
    const float* w_scales, const int32_t* bias, int8_t* out, float out_scale,
    int32_t out_zp, uint32_t batch, uint32_t in_features,
    uint32_t out_features, int activation);

/**
 * TFLite-compatible int8 2D convolution. in: NHWC (in_scale, in_zp);
 * filter: OHWI int8, symmetric, with `n_filter_scales` scales (1 for
 * per-tensor or out_c for per-channel); bias: [out_c] int32 or NULL;
 * out: NHWC (out_scale, out_zp).
 */
PTENSOR_C_EXPORT int ptensor_conv2d_i8(
    const int8_t* in, uint32_t n, uint32_t h, uint32_t w, uint32_t c,
    float in_scale, int32_t in_zp, const int8_t* filter, uint32_t out_c,
    uint32_t kh, uint32_t kw, const float* filter_scales,
    uint32_t n_filter_scales, const int32_t* bias, uint32_t stride_h,
    uint32_t stride_w, int padding, int activation, int8_t* out,
    float out_scale, int32_t out_zp);

/** int8 max pooling, NHWC. Input and output share quantization params. */
PTENSOR_C_EXPORT int ptensor_maxpool2d_i8(const int8_t* in, uint32_t n,
                                          uint32_t h, uint32_t w, uint32_t c,
                                          uint32_t kh, uint32_t kw,
                                          uint32_t stride_h, uint32_t stride_w,
                                          int padding, int8_t* out);

/* ---- packed storage --------------------------------------------------- */

/** Bytes needed to store `count` values in the given packed format. */
PTENSOR_C_EXPORT size_t ptensor_packed_size(int format, uint32_t bits,
                                            uint32_t count);
/** Pack `count` unsigned codes of `bits` bits (1..16) into `out`. */
PTENSOR_C_EXPORT int ptensor_pack_bits(const uint32_t* codes, uint32_t count,
                                       uint32_t bits, uint8_t* out);
/**
 * Decode `count` packed values to float exactly as a PackedRomTensor with a
 * float logical type would present them to an operator.
 */
PTENSOR_C_EXPORT int ptensor_decode_packed_f32(int format, uint32_t bits,
                                               float scale, int32_t zero_point,
                                               const void* data, uint32_t count,
                                               float* out);
PTENSOR_C_EXPORT uint16_t ptensor_float_to_half(float value);
PTENSOR_C_EXPORT float ptensor_half_to_float(uint16_t value);

#ifdef __cplusplus
}
#endif
#endif
