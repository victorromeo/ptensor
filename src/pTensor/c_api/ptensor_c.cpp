#include "ptensor_c.h"

#include <cmath>
#include <cstring>

#include "pTensor/allocators/arenaAllocator.hpp"
#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/ops/reference/AddOperator.hpp"
#include "pTensor/ops/reference/AvgPoolOperator.hpp"
#include "pTensor/ops/reference/ConvolutionOperator.hpp"
#include "pTensor/ops/reference/FullyConnectedOperator.hpp"
#include "pTensor/ops/reference/LogisticOperator.hpp"
#include "pTensor/ops/reference/MatrixMultiplyOperator.hpp"
#include "pTensor/ops/reference/MaxPoolOperator.hpp"
#include "pTensor/ops/reference/ReLuOperator.hpp"
#include "pTensor/ops/reference/SoftMaxOperator.hpp"
#include "pTensor/ops/reference/TanhOperator.hpp"
#include "pTensor/ptensor_config.h"
#include "pTensor/tensors/BufferTensor.hpp"
#include "pTensor/tensors/PackedRomTensor.hpp"
#include "pTensor/tensors/RomTensor.hpp"
#include "pTensor/util/packing.hpp"
#if PTENSOR_WITH_QUANTIZED_OPS
#include "pTensor/ops/tflm/DequantizeOperator.hpp"
#include "pTensor/ops/tflm/QuantizeOperator.hpp"
#include "pTensor/ops/tflm/fully_connected.hpp"
#endif

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

uint32_t g_last_error = 0;

class RecordingErrorHandler : public ErrorHandler {
 public:
  bool failed = false;
  uint32_t last = 0;
  virtual void uThrow(Error* err) override {
    failed = true;
    last = err ? err->event_id : 0;
  }
  virtual void notify(const Event&) override {}
};

// Allocators used for tensor metadata and operator scratch during one call.
// Caller buffers are wrapped, never copied, so these stay small.
localCircularArenaAllocator<16 * 1024, uint32_t, 128> g_meta;
localCircularArenaAllocator<64 * 1024, uint32_t, 64> g_ram;

// Installs the C API's allocators and error handler on the default context for
// the duration of one call, then restores whatever was there before.
class Session {
 public:
  Session() {
    Context* ctx = Context::get_default_context();
    _prev_handler = ctx->get_default_error_handler();
    ctx->set_ErrorHandler(&_handler);
    // Context only exposes getters that raise when unset; peek safely.
    _prev_meta = ctx->peek_metadata_allocator();
    _prev_ram = ctx->peek_ram_data_allocator();
    ctx->set_metadata_allocator(&g_meta);
    ctx->set_ram_data_allocator(&g_ram);
  }
  ~Session() {
    g_meta.clear();
    g_ram.clear();
    Context* ctx = Context::get_default_context();
    ctx->set_metadata_allocator(_prev_meta);
    ctx->set_ram_data_allocator(_prev_ram);
    ctx->set_ErrorHandler(_prev_handler);
  }
  int status() {
    if (_handler.failed) {
      g_last_error = _handler.last;
      return PTENSOR_ERR_RUNTIME;
    }
    g_last_error = 0;
    return PTENSOR_OK;
  }

 private:
  RecordingErrorHandler _handler;
  ErrorHandler* _prev_handler;
  AllocatorInterface* _prev_meta;
  AllocatorInterface* _prev_ram;
};

int invalid() {
  g_last_error = 0;
  return PTENSOR_ERR_INVALID_ARGUMENT;
}

bool fits_dim(uint32_t v) { return v > 0 && v <= 0xFFFFu; }
bool valid_padding(int p) {
  return p == PTENSOR_PADDING_VALID || p == PTENSOR_PADDING_SAME;
}

Fuseable::Activation<float> float_activation(int act) {
  switch (act) {
    case PTENSOR_ACT_RELU:
      return Fuseable::ReLU<float>;
    case PTENSOR_ACT_RELU6:
      return Fuseable::ReLU6<float>;
    default:
      return Fuseable::NoActivation<float>;
  }
}

bool valid_activation(int act) {
  return act == PTENSOR_ACT_NONE || act == PTENSOR_ACT_RELU ||
         act == PTENSOR_ACT_RELU6;
}

template <typename Op>
int run_unary_f32(const float* in, float* out, uint32_t count) {
  if (!in || !out || !fits_dim(count)) return invalid();
  Session s;
  {
    Tensor tin = new RomTensor({static_cast<uint16_t>(count)}, flt, in);
    Tensor tout = new BufferTensor({static_cast<uint16_t>(count)}, flt, out);
    Op op;
    op.set_inputs({{Op::in, tin}}).set_outputs({{Op::out, tout}}).eval();
  }
  return s.status();
}

template <typename PoolOp>
int run_pool_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                 uint32_t c, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
                 int padding, float* out) {
  if (!in || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) ||
      !fits_dim(c) || !fits_dim(kh) || !fits_dim(kw) || !fits_dim(sh) ||
      !fits_dim(sw) || !valid_padding(padding))
    return invalid();
  const uint32_t oh = ptensor_conv_output_size(h, kh, sh, padding);
  const uint32_t ow = ptensor_conv_output_size(w, kw, sw, padding);
  if (!oh || !ow) return invalid();
  Session s;
  {
    Tensor tin = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, flt, in);
    Tensor tout = new BufferTensor({(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)c}, flt, out);
    PoolOp op({(uint16_t)kh, (uint16_t)kw}, {1, (uint16_t)sh, (uint16_t)sw, 1},
              static_cast<Padding>(padding));
    op.set_inputs({{PoolOp::in, tin}}).set_outputs({{PoolOp::out, tout}}).eval();
  }
  return s.status();
}

}  // namespace

extern "C" {

const char* ptensor_version(void) { return "1.0.0"; }
uint32_t ptensor_last_error(void) { return g_last_error; }

uint32_t ptensor_conv_output_size(uint32_t in_size, uint32_t kernel,
                                  uint32_t stride, int padding) {
  if (!in_size || !kernel || !stride) return 0;
  if (padding == PTENSOR_PADDING_SAME) return (in_size + stride - 1) / stride;
  if (padding == PTENSOR_PADDING_VALID) {
    if (kernel > in_size) return 0;
    return (in_size - kernel) / stride + 1;
  }
  return 0;
}

int ptensor_matmul_f32(const float* a, const float* b, float* out, uint32_t m,
                       uint32_t k, uint32_t n) {
  if (!a || !b || !out || !fits_dim(m) || !fits_dim(k) || !fits_dim(n))
    return invalid();
  Session s;
  {
    Tensor ta = new RomTensor({(uint16_t)m, (uint16_t)k}, flt, a);
    Tensor tb = new RomTensor({(uint16_t)k, (uint16_t)n}, flt, b);
    Tensor tc = new BufferTensor({(uint16_t)m, (uint16_t)n}, flt, out);
    MatrixMultOperator<float> op;
    op.set_inputs({{MatrixMultOperator<float>::a, ta},
                   {MatrixMultOperator<float>::b, tb}})
        .set_outputs({{MatrixMultOperator<float>::c, tc}})
        .eval();
  }
  return s.status();
}

int ptensor_fully_connected_f32(const float* in, const float* weights,
                                const float* bias, float* out, uint32_t batch,
                                uint32_t in_features, uint32_t out_features,
                                int activation) {
  if (!in || !weights || !out || !fits_dim(batch) || !fits_dim(in_features) ||
      !fits_dim(out_features) || !valid_activation(activation))
    return invalid();
  Session s;
  {
    typedef FullyConnectedOperator<float> FC;
    Tensor tin = new RomTensor({(uint16_t)batch, (uint16_t)in_features}, flt, in);
    Tensor tw = new RomTensor({(uint16_t)in_features, (uint16_t)out_features}, flt, weights);
    Tensor tout = new BufferTensor({(uint16_t)batch, (uint16_t)out_features}, flt, out);
    FC op(float_activation(activation));
    if (bias) {
      Tensor tb = new RomTensor({(uint16_t)out_features}, flt, bias);
      op.set_inputs({{FC::input, tin}, {FC::filter, tw}, {FC::bias, tb}})
          .set_outputs({{FC::output, tout}})
          .eval();
    } else {
      op.set_inputs({{FC::input, tin}, {FC::filter, tw}})
          .set_outputs({{FC::output, tout}})
          .eval();
    }
  }
  return s.status();
}

int ptensor_add_f32(const float* a, const float* b, float* out,
                    uint32_t count) {
  if (!a || !b || !out || !fits_dim(count)) return invalid();
  Session s;
  {
    Tensor ta = new RomTensor({(uint16_t)count}, flt, a);
    Tensor tb = new RomTensor({(uint16_t)count}, flt, b);
    Tensor tc = new BufferTensor({(uint16_t)count}, flt, out);
    AddOperator<float> op;
    op.set_inputs({{AddOperator<float>::a, ta}, {AddOperator<float>::b, tb}})
        .set_outputs({{AddOperator<float>::c, tc}})
        .eval();
  }
  return s.status();
}

int ptensor_relu_f32(const float* in, float* out, uint32_t count) {
  return run_unary_f32<ReLUOperator<float> >(in, out, count);
}
int ptensor_relu6_f32(const float* in, float* out, uint32_t count) {
  return run_unary_f32<ReLU6Operator<float> >(in, out, count);
}
int ptensor_sigmoid_f32(const float* in, float* out, uint32_t count) {
  return run_unary_f32<LogisticOperator<float> >(in, out, count);
}
int ptensor_tanh_f32(const float* in, float* out, uint32_t count) {
  if (!in || !out || !fits_dim(count)) return invalid();
  Session s;
  {
    typedef TanhOperator<float> Op;
    Tensor tin = new RomTensor({(uint16_t)count}, flt, in);
    Tensor tout = new BufferTensor({(uint16_t)count}, flt, out);
    Op op;
    op.set_inputs({{Op::act_in, tin}}).set_outputs({{Op::act_out, tout}}).eval();
  }
  return s.status();
}

int ptensor_softmax_f32(const float* in, float* out, uint32_t rows,
                        uint32_t depth) {
  if (!in || !out || !fits_dim(rows) || !fits_dim(depth)) return invalid();
  Session s;
  {
    typedef SoftmaxOperator<float> Op;
    Tensor tin = new RomTensor({(uint16_t)rows, (uint16_t)depth}, flt, in);
    Tensor tout = new BufferTensor({(uint16_t)rows, (uint16_t)depth}, flt, out);
    Op op;
    op.set_inputs({{Op::in, tin}}).set_outputs({{Op::out, tout}}).eval();
  }
  return s.status();
}

int ptensor_conv2d_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                       uint32_t c, const float* filter, uint32_t out_c,
                       uint32_t kh, uint32_t kw, const float* bias,
                       uint32_t stride_h, uint32_t stride_w, int padding,
                       float* out) {
  if (!in || !filter || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) ||
      !fits_dim(c) || !fits_dim(out_c) || !fits_dim(kh) || !fits_dim(kw) ||
      !fits_dim(stride_h) || !fits_dim(stride_w) || !valid_padding(padding))
    return invalid();
  const uint32_t oh = ptensor_conv_output_size(h, kh, stride_h, padding);
  const uint32_t ow = ptensor_conv_output_size(w, kw, stride_w, padding);
  if (!oh || !ow) return invalid();
  Session s;
  {
    typedef Conv2dOperator<float> Op;
    Tensor tin = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, flt, in);
    Tensor tf = new RomTensor({(uint16_t)out_c, (uint16_t)kh, (uint16_t)kw, (uint16_t)c}, flt, filter);
    Tensor tout = new BufferTensor({(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)out_c}, flt, out);
    Op op({(uint16_t)stride_h, (uint16_t)stride_w}, static_cast<Padding>(padding));
    if (bias) {
      Tensor tb = new RomTensor({(uint16_t)out_c}, flt, bias);
      op.set_inputs({{Op::in, tin}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, tout}})
          .eval();
    } else {
      op.set_inputs({{Op::in, tin}, {Op::filter, tf}})
          .set_outputs({{Op::out, tout}})
          .eval();
    }
  }
  return s.status();
}

int ptensor_maxpool2d_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                          uint32_t c, uint32_t kh, uint32_t kw,
                          uint32_t stride_h, uint32_t stride_w, int padding,
                          float* out) {
  return run_pool_f32<MaxPoolOperator<float> >(in, n, h, w, c, kh, kw,
                                               stride_h, stride_w, padding, out);
}
int ptensor_avgpool2d_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                          uint32_t c, uint32_t kh, uint32_t kw,
                          uint32_t stride_h, uint32_t stride_w, int padding,
                          float* out) {
  return run_pool_f32<AvgPoolOperator<float> >(in, n, h, w, c, kh, kw,
                                               stride_h, stride_w, padding, out);
}

#if PTENSOR_WITH_QUANTIZED_OPS
int ptensor_quantize_i8(const float* in, int8_t* out, uint32_t count,
                        float scale, int32_t zero_point) {
  if (!in || !out || !fits_dim(count) || !(scale > 0)) return invalid();
  Session s;
  {
    typedef TflmSymQuantOps::QuantizeOperator<int8_t, float> Op;
    Tensor tin = new RomTensor({(uint16_t)count}, flt, in);
    Tensor tout = new BufferTensor({(uint16_t)count}, i8, out);
    tout->set_quantization_params(PerTensorQuantizationParams(zero_point, scale));
    Op op;
    op.set_inputs({{Op::input, tin}}).set_outputs({{Op::output, tout}}).eval();
  }
  return s.status();
}

int ptensor_dequantize_i8(const int8_t* in, float* out, uint32_t count,
                          float scale, int32_t zero_point) {
  if (!in || !out || !fits_dim(count) || !(scale > 0)) return invalid();
  Session s;
  {
    typedef TflmSymQuantOps::DequantizeOperator<float, int8_t> Op;
    Tensor tin = new RomTensor({(uint16_t)count}, i8, in);
    tin->set_quantization_params(PerTensorQuantizationParams(zero_point, scale));
    Tensor tout = new BufferTensor({(uint16_t)count}, flt, out);
    Op op;
    op.set_inputs({{Op::a, tin}}).set_outputs({{Op::b, tout}}).eval();
  }
  return s.status();
}

int ptensor_fully_connected_i8(const int8_t* in, float in_scale, int32_t in_zp,
                               const int8_t* weights, float w_scale,
                               const int32_t* bias, int8_t* out,
                               float out_scale, int32_t out_zp, uint32_t batch,
                               uint32_t in_features, uint32_t out_features,
                               int activation) {
  if (!in || !weights || !out || !fits_dim(batch) || !fits_dim(in_features) ||
      !fits_dim(out_features) || !valid_activation(activation) ||
      !(in_scale > 0) || !(w_scale > 0) || !(out_scale > 0))
    return invalid();
  Session s;
  {
    typedef TflmSymQuantOps::FullyConnectedOperator<int8_t> FC;
    Tensor tin = new RomTensor({(uint16_t)batch, (uint16_t)in_features}, i8, in);
    tin->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
    Tensor tw = new RomTensor({(uint16_t)in_features, (uint16_t)out_features}, i8, weights);
    tw->set_quantization_params(PerTensorQuantizationParams(0, w_scale));
    Tensor tout = new BufferTensor({(uint16_t)batch, (uint16_t)out_features}, i8, out);
    tout->set_quantization_params(PerTensorQuantizationParams(out_zp, out_scale));
    FC op(static_cast<TFLM::TfLiteFusedActivation>(activation));
    if (bias) {
      Tensor tb = new RomTensor({(uint16_t)out_features}, i32, bias);
      tb->set_quantization_params(PerTensorQuantizationParams(0, in_scale * w_scale));
      op.set_inputs({{FC::input, tin}, {FC::filter, tw}, {FC::bias, tb}})
          .set_outputs({{FC::output, tout}})
          .eval();
    } else {
      op.set_inputs({{FC::input, tin}, {FC::filter, tw}})
          .set_outputs({{FC::output, tout}})
          .eval();
    }
  }
  return s.status();
}
#else
int ptensor_quantize_i8(const float*, int8_t*, uint32_t, float, int32_t) {
  return invalid();
}
int ptensor_dequantize_i8(const int8_t*, float*, uint32_t, float, int32_t) {
  return invalid();
}
int ptensor_fully_connected_i8(const int8_t*, float, int32_t, const int8_t*,
                               float, const int32_t*, int8_t*, float, int32_t,
                               uint32_t, uint32_t, uint32_t, int) {
  return invalid();
}
#endif

size_t ptensor_packed_size(int format, uint32_t bits, uint32_t count) {
  if (format < PTENSOR_PACKED_UINT_BITS || format > PTENSOR_PACKED_FLOAT64)
    return 0;
  PackedEncoding e = PackedEncoding::of(static_cast<PackedFormat>(format));
  e.bit_width = static_cast<uint8_t>(bits);
  if (format == PTENSOR_PACKED_UINT_BITS && (bits < 1 || bits > 16)) return 0;
  return packed_buffer_size(e, count);
}

int ptensor_pack_bits(const uint32_t* codes, uint32_t count, uint32_t bits,
                      uint8_t* out) {
  if (!codes || !out || bits < 1 || bits > 16) return invalid();
  const uint32_t max_code = (1u << bits) - 1u;
  for (uint32_t i = 0; i < count; ++i) {
    if (codes[i] > max_code) return invalid();
  }
  packing::pack_bits(codes, count, static_cast<uint8_t>(bits), out);
  g_last_error = 0;
  return PTENSOR_OK;
}

int ptensor_decode_packed_f32(int format, uint32_t bits, float scale,
                              int32_t zero_point, const void* data,
                              uint32_t count, float* out) {
  if (!data || !out || !fits_dim(count) ||
      ptensor_packed_size(format, bits, count) == 0)
    return invalid();
  Session s;
  {
    PackedEncoding e = PackedEncoding::of(static_cast<PackedFormat>(format),
                                          scale, zero_point);
    e.bit_width = static_cast<uint8_t>(bits);
    Tensor t = new PackedRomTensor({(uint16_t)count}, flt, e, data);
    const Tensor& ct = t;
    for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(ct(i));
  }
  return s.status();
}

uint16_t ptensor_float_to_half(float value) {
  return packing::float_to_half(value);
}
float ptensor_half_to_float(uint16_t value) {
  return packing::half_to_float(value);
}

}  // extern "C"
