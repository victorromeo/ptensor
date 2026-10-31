#include "ptensor_c.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "pTensor/allocators/arenaAllocator.hpp"
#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/ops/reference/AddOperator.hpp"
#include "pTensor/ops/reference/AvgPoolOperator.hpp"
#include "pTensor/ops/reference/ConvolutionOperator.hpp"
#include "pTensor/ops/reference/ElementwiseOperators.hpp"
#include "pTensor/ops/reference/LayerOperators.hpp"
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
#include "pTensor/ops/tflm/convolution2d.hpp"
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

template <typename PoolOp, typename T>
int run_pool(const T* in, ttype type, uint32_t n, uint32_t h, uint32_t w,
             uint32_t c, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
             int padding, T* out) {
  if (!in || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) ||
      !fits_dim(c) || !fits_dim(kh) || !fits_dim(kw) || !fits_dim(sh) ||
      !fits_dim(sw) || !valid_padding(padding))
    return invalid();
  const uint32_t oh = ptensor_conv_output_size(h, kh, sh, padding);
  const uint32_t ow = ptensor_conv_output_size(w, kw, sw, padding);
  if (!oh || !ow) return invalid();
  Session s;
  {
    Tensor tin = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, type, in);
    Tensor tout = new BufferTensor({(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)c}, type, out);
    PoolOp op({(uint16_t)kh, (uint16_t)kw}, {1, (uint16_t)sh, (uint16_t)sw, 1},
              static_cast<Padding>(padding));
    op.set_inputs({{PoolOp::in, tin}}).set_outputs({{PoolOp::out, tout}}).eval();
  }
  return s.status();
}

bool make_shape(const uint32_t* dims, uint32_t rank, TensorShape& out) {
  if (!dims || rank < 1 || rank > 4) return false;
  uint16_t d[4] = {0, 0, 0, 0};
  for (uint32_t i = 0; i < rank; ++i) {
    if (!fits_dim(dims[i])) return false;
    d[i] = static_cast<uint16_t>(dims[i]);
  }
  switch (rank) {
    case 1: out = TensorShape(d[0]); break;
    case 2: out = TensorShape(d[0], d[1]); break;
    case 3: out = TensorShape(d[0], d[1], d[2]); break;
    default: out = TensorShape(d[0], d[1], d[2], d[3]); break;
  }
  return true;
}

uint32_t elems(const uint32_t* dims, uint32_t rank) {
  uint32_t n = 1;
  for (uint32_t i = 0; i < rank; ++i) n *= dims[i];
  return n;
}

template <typename Fn>
int run_binary(const float* a, const TensorShape& sa, const float* b,
               const TensorShape& sb, const TensorShape& so, float* out) {
  Session s;
  {
    typedef BinaryElementwiseOperator<float, Fn> Op;
    Tensor ta = new RomTensor(sa, flt, a);
    Tensor tb = new RomTensor(sb, flt, b);
    Tensor tc = new BufferTensor(so, flt, out);
    Op op;
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::c, tc}}).eval();
  }
  return s.status();
}

template <typename Fn>
int run_unary(const float* in, float* out, uint32_t count, const Fn& fn) {
  Session s;
  {
    typedef UnaryElementwiseOperator<float, Fn> Op;
    Tensor ti = new RomTensor({(uint16_t)count}, flt, in);
    Tensor to = new BufferTensor({(uint16_t)count}, flt, out);
    Op op(fn);
    op.set_inputs({{Op::in, ti}}).set_outputs({{Op::out, to}}).eval();
  }
  return s.status();
}

// pTensor dimensions are uint16_t, so a flat elementwise call over more than
// 65535 values is split into chunks. `f(offset, n)` processes one chunk.
const uint32_t kMaxDim = 0xFFFFu;
template <typename F>
int chunked(uint32_t count, uint32_t granule, const F& f) {
  const uint32_t step = (kMaxDim / granule) * granule;
  for (uint32_t off = 0; off < count; off += step) {
    const uint32_t n = count - off < step ? count - off : step;
    const int st = f(off, n);
    if (st != PTENSOR_OK) return st;
  }
  return PTENSOR_OK;
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
  if (a && b && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_add_f32(a + o, b + o, out + o, n);
    });
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
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_relu_f32(in + o, out + o, n);
    });
  return run_unary_f32<ReLUOperator<float> >(in, out, count);
}
int ptensor_relu6_f32(const float* in, float* out, uint32_t count) {
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_relu6_f32(in + o, out + o, n);
    });
  return run_unary_f32<ReLU6Operator<float> >(in, out, count);
}
int ptensor_sigmoid_f32(const float* in, float* out, uint32_t count) {
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_sigmoid_f32(in + o, out + o, n);
    });
  return run_unary_f32<LogisticOperator<float> >(in, out, count);
}
int ptensor_tanh_f32(const float* in, float* out, uint32_t count) {
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_tanh_f32(in + o, out + o, n);
    });
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
  return run_pool<MaxPoolOperator<float> >(in, flt, n, h, w, c, kh, kw,
                                           stride_h, stride_w, padding, out);
}
int ptensor_maxpool2d_i8(const int8_t* in, uint32_t n, uint32_t h, uint32_t w,
                         uint32_t c, uint32_t kh, uint32_t kw,
                         uint32_t stride_h, uint32_t stride_w, int padding,
                         int8_t* out) {
  // Max is monotonic, so int8 pooling needs no requantization.
  return run_pool<MaxPoolOperator<int8_t> >(in, i8, n, h, w, c, kh, kw,
                                            stride_h, stride_w, padding, out);
}
int ptensor_avgpool2d_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                          uint32_t c, uint32_t kh, uint32_t kw,
                          uint32_t stride_h, uint32_t stride_w, int padding,
                          float* out) {
  return run_pool<AvgPoolOperator<float> >(in, flt, n, h, w, c, kh, kw,
                                           stride_h, stride_w, padding, out);
}

int ptensor_binary_f32(int op, const float* a, const uint32_t* a_shape,
                       uint32_t a_rank, const float* b, const uint32_t* b_shape,
                       uint32_t b_rank, float* out) {
  TensorShape sa(1), sb(1);
  if (!a || !b || !out || !make_shape(a_shape, a_rank, sa) ||
      !make_shape(b_shape, b_rank, sb))
    return invalid();
  // NumPy broadcast of the two shapes, right aligned.
  const uint32_t rank = a_rank > b_rank ? a_rank : b_rank;
  uint32_t od[4];
  for (uint32_t i = 0; i < rank; ++i) {
    const int32_t ia = static_cast<int32_t>(i) - static_cast<int32_t>(rank - a_rank);
    const int32_t ib = static_cast<int32_t>(i) - static_cast<int32_t>(rank - b_rank);
    const uint32_t da = ia >= 0 ? a_shape[ia] : 1;
    const uint32_t db = ib >= 0 ? b_shape[ib] : 1;
    if (da != db && da != 1 && db != 1) return invalid();
    od[i] = da > db ? da : db;
  }
  TensorShape so(1);
  if (!make_shape(od, rank, so)) return invalid();
  using namespace elementwise;
  switch (op) {
    case PTENSOR_BINARY_ADD: return run_binary<Add<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_SUB: return run_binary<Sub<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_MUL: return run_binary<Mul<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_DIV: return run_binary<Div<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_MAXIMUM: return run_binary<Maximum<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_MINIMUM: return run_binary<Minimum<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_SQUARED_DIFFERENCE:
      return run_binary<SquaredDifference<float> >(a, sa, b, sb, so, out);
    case PTENSOR_BINARY_POW: return run_binary<Pow<float> >(a, sa, b, sb, so, out);
    default: return invalid();
  }
}

int ptensor_unary_f32(int op, const float* in, float* out, uint32_t count,
                      float p0, float p1) {
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_unary_f32(op, in + o, out + o, n, p0, p1);
    });
  if (!in || !out || !fits_dim(count)) return invalid();
  using namespace activations;
  switch (op) {
    case PTENSOR_UNARY_LEAKY_RELU: return run_unary(in, out, count, LeakyReLU(p0));
    case PTENSOR_UNARY_ELU: return run_unary(in, out, count, ELU(p0));
    case PTENSOR_UNARY_SELU: return run_unary(in, out, count, SELU());
    case PTENSOR_UNARY_HARD_SIGMOID: return run_unary(in, out, count, HardSigmoid());
    case PTENSOR_UNARY_HARD_SWISH: return run_unary(in, out, count, HardSwish());
    case PTENSOR_UNARY_SWISH: return run_unary(in, out, count, Swish());
    case PTENSOR_UNARY_GELU: return run_unary(in, out, count, GELU());
    case PTENSOR_UNARY_SOFTPLUS: return run_unary(in, out, count, Softplus());
    case PTENSOR_UNARY_SOFTSIGN: return run_unary(in, out, count, Softsign());
    case PTENSOR_UNARY_ABS: return run_unary(in, out, count, Abs());
    case PTENSOR_UNARY_NEG: return run_unary(in, out, count, Neg());
    case PTENSOR_UNARY_SQUARE: return run_unary(in, out, count, Square());
    case PTENSOR_UNARY_SQRT: return run_unary(in, out, count, Sqrt());
    case PTENSOR_UNARY_RSQRT: return run_unary(in, out, count, Rsqrt());
    case PTENSOR_UNARY_EXP: return run_unary(in, out, count, Exp());
    case PTENSOR_UNARY_LOG: return run_unary(in, out, count, Log());
    case PTENSOR_UNARY_FLOOR: return run_unary(in, out, count, Floor());
    case PTENSOR_UNARY_CEIL: return run_unary(in, out, count, Ceil());
    case PTENSOR_UNARY_ROUND: return run_unary(in, out, count, Round());
    case PTENSOR_UNARY_SIN: return run_unary(in, out, count, Sin());
    case PTENSOR_UNARY_COS: return run_unary(in, out, count, Cos());
    case PTENSOR_UNARY_CLIP:
      if (!(p0 <= p1)) return invalid();
      return run_unary(in, out, count, Clip(p0, p1));
    default: return invalid();
  }
}

int ptensor_batchnorm_f32(const float* in, uint32_t count, uint32_t channels,
                          const float* mean, const float* variance,
                          const float* gamma, const float* beta, float epsilon,
                          float* out) {
  if (in && out && channels && fits_dim(channels) && count % channels == 0 &&
      count / channels > kMaxDim)
    return chunked(count, channels, [&](uint32_t o, uint32_t n) {
      return ptensor_batchnorm_f32(in + o, n, channels, mean, variance, gamma,
                                   beta, epsilon, out + o);
    });
  if (!in || !mean || !variance || !gamma || !beta || !out ||
      !fits_dim(channels) || count % channels != 0 ||
      !fits_dim(count / channels) || !(epsilon >= 0))
    return invalid();
  Session s;
  {
    typedef BatchNormOperator<float> Op;
    const TensorShape shape((uint16_t)(count / channels), (uint16_t)channels);
    Tensor ti = new RomTensor(shape, flt, in);
    Tensor tm = new RomTensor({(uint16_t)channels}, flt, mean);
    Tensor tv = new RomTensor({(uint16_t)channels}, flt, variance);
    Tensor tg = new RomTensor({(uint16_t)channels}, flt, gamma);
    Tensor tb = new RomTensor({(uint16_t)channels}, flt, beta);
    Tensor to = new BufferTensor(shape, flt, out);
    Op op(epsilon);
    op.set_inputs({{Op::in, ti}, {Op::mean, tm}, {Op::variance, tv},
                   {Op::gamma, tg}, {Op::beta, tb}})
        .set_outputs({{Op::out, to}})
        .eval();
  }
  return s.status();
}

int ptensor_pad2d_f32(const float* in, uint32_t n, uint32_t h, uint32_t w,
                      uint32_t c, uint32_t top, uint32_t bottom, uint32_t left,
                      uint32_t right, float value, float* out) {
  if (!in || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) ||
      !fits_dim(c) || !fits_dim(h + top + bottom) || !fits_dim(w + left + right))
    return invalid();
  Session s;
  {
    typedef PadOperator<float> Op;
    Tensor ti = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, flt, in);
    Tensor to = new BufferTensor({(uint16_t)n, (uint16_t)(h + top + bottom),
                                  (uint16_t)(w + left + right), (uint16_t)c}, flt, out);
    Op op((uint16_t)top, (uint16_t)bottom, (uint16_t)left, (uint16_t)right, value);
    op.set_inputs({{Op::in, ti}}).set_outputs({{Op::out, to}}).eval();
  }
  return s.status();
}

int ptensor_concat_f32(const float* a, const uint32_t* a_shape, const float* b,
                       const uint32_t* b_shape, uint32_t rank, int axis,
                       float* out) {
  TensorShape sa(1), sb(1), so(1);
  if (!a || !b || !out || !make_shape(a_shape, rank, sa) ||
      !make_shape(b_shape, rank, sb))
    return invalid();
  const int ax = axis < 0 ? axis + static_cast<int>(rank) : axis;
  if (ax < 0 || ax >= static_cast<int>(rank)) return invalid();
  uint32_t od[4];
  for (uint32_t i = 0; i < rank; ++i) {
    if (static_cast<int>(i) == ax) {
      od[i] = a_shape[i] + b_shape[i];
    } else {
      if (a_shape[i] != b_shape[i]) return invalid();
      od[i] = a_shape[i];
    }
  }
  if (!make_shape(od, rank, so)) return invalid();
  Session s;
  {
    typedef ConcatOperator<float> Op;
    Tensor ta = new RomTensor(sa, flt, a);
    Tensor tb = new RomTensor(sb, flt, b);
    Tensor to = new BufferTensor(so, flt, out);
    Op op(static_cast<int8_t>(ax));
    op.set_inputs({{Op::a, ta}, {Op::b, tb}}).set_outputs({{Op::out, to}}).eval();
  }
  return s.status();
}

int ptensor_global_avgpool_f32(const float* in, uint32_t n, uint32_t h,
                               uint32_t w, uint32_t c, float* out) {
  if (!in || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) || !fits_dim(c))
    return invalid();
  Session s;
  {
    typedef GlobalAveragePoolOperator<float> Op;
    Tensor ti = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, flt, in);
    Tensor to = new BufferTensor({(uint16_t)n, (uint16_t)c}, flt, out);
    Op op;
    op.set_inputs({{Op::in, ti}}).set_outputs({{Op::out, to}}).eval();
  }
  return s.status();
}

int ptensor_depthwise_conv2d_f32(const float* in, uint32_t n, uint32_t h,
                                 uint32_t w, uint32_t c, const float* filter,
                                 uint32_t kh, uint32_t kw, uint32_t multiplier,
                                 const float* bias, uint32_t stride_h,
                                 uint32_t stride_w, int padding,
                                 uint32_t dilation_h, uint32_t dilation_w,
                                 float* out) {
  if (!in || !filter || !out || !fits_dim(n) || !fits_dim(h) || !fits_dim(w) ||
      !fits_dim(c) || !fits_dim(kh) || !fits_dim(kw) || !fits_dim(multiplier) ||
      !fits_dim(c * multiplier) || !fits_dim(stride_h) || !fits_dim(stride_w) ||
      !fits_dim(dilation_h) || !fits_dim(dilation_w) || !valid_padding(padding))
    return invalid();
  const uint32_t ekh = (kh - 1) * dilation_h + 1, ekw = (kw - 1) * dilation_w + 1;
  const uint32_t oh = ptensor_conv_output_size(h, ekh, stride_h, padding);
  const uint32_t ow = ptensor_conv_output_size(w, ekw, stride_w, padding);
  if (!oh || !ow) return invalid();
  const uint16_t oc = static_cast<uint16_t>(c * multiplier);
  Session s;
  {
    typedef DepthwiseConv2dOperator<float> Op;
    Tensor ti = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, flt, in);
    Tensor tf = new RomTensor({1, (uint16_t)kh, (uint16_t)kw, oc}, flt, filter);
    Tensor to = new BufferTensor({(uint16_t)n, (uint16_t)oh, (uint16_t)ow, oc}, flt, out);
    Op op((uint16_t)stride_h, (uint16_t)stride_w, static_cast<Padding>(padding),
          (uint16_t)multiplier, (uint16_t)dilation_h, (uint16_t)dilation_w);
    if (bias) {
      Tensor tb = new RomTensor({oc}, flt, bias);
      op.set_inputs({{Op::in, ti}, {Op::filter, tf}, {Op::bias, tb}})
          .set_outputs({{Op::out, to}})
          .eval();
    } else {
      op.set_inputs({{Op::in, ti}, {Op::filter, tf}})
          .set_outputs({{Op::out, to}})
          .eval();
    }
  }
  return s.status();
}

int ptensor_lstm_cell_f32(const float* x, const float* h, const float* c,
                          const float* W, const float* U, const float* b,
                          uint32_t batch, uint32_t in_features, uint32_t hidden,
                          float* h_out, float* c_out) {
  if (!x || !h || !c || !W || !U || !b || !h_out || !c_out || !fits_dim(batch) ||
      !fits_dim(in_features) || !fits_dim(hidden) || !fits_dim(4 * hidden))
    return invalid();
  Session s;
  {
    typedef LstmCellOperator Op;
    const uint16_t B = (uint16_t)batch, I = (uint16_t)in_features,
                   H = (uint16_t)hidden, H4 = (uint16_t)(4 * hidden);
    Tensor tx = new RomTensor({B, I}, flt, x);
    Tensor th = new RomTensor({B, H}, flt, h);
    Tensor tc = new RomTensor({B, H}, flt, c);
    Tensor tW = new RomTensor({I, H4}, flt, W);
    Tensor tU = new RomTensor({H, H4}, flt, U);
    Tensor tb = new RomTensor({H4}, flt, b);
    Tensor tho = new BufferTensor({B, H}, flt, h_out);
    Tensor tco = new BufferTensor({B, H}, flt, c_out);
    Op op;
    op.set_inputs({{Op::x, tx}, {Op::h, th}, {Op::c, tc}, {Op::W, tW},
                   {Op::U, tU}, {Op::b, tb}})
        .set_outputs({{Op::h_out, tho}, {Op::c_out, tco}})
        .eval();
  }
  return s.status();
}

int ptensor_gru_cell_f32(const float* x, const float* h, const float* W,
                         const float* U, const float* bx, const float* bh,
                         uint32_t batch, uint32_t in_features, uint32_t hidden,
                         float* h_out) {
  if (!x || !h || !W || !U || !bx || !bh || !h_out || !fits_dim(batch) ||
      !fits_dim(in_features) || !fits_dim(hidden) || !fits_dim(3 * hidden))
    return invalid();
  Session s;
  {
    typedef GruCellOperator Op;
    const uint16_t B = (uint16_t)batch, I = (uint16_t)in_features,
                   H = (uint16_t)hidden, H3 = (uint16_t)(3 * hidden);
    Tensor tx = new RomTensor({B, I}, flt, x);
    Tensor th = new RomTensor({B, H}, flt, h);
    Tensor tW = new RomTensor({I, H3}, flt, W);
    Tensor tU = new RomTensor({H, H3}, flt, U);
    Tensor tbx = new RomTensor({H3}, flt, bx);
    Tensor tbh = new RomTensor({H3}, flt, bh);
    Tensor tho = new BufferTensor({B, H}, flt, h_out);
    Op op;
    op.set_inputs({{Op::x, tx}, {Op::h, th}, {Op::W, tW}, {Op::U, tU},
                   {Op::bx, tbx}, {Op::bh, tbh}})
        .set_outputs({{Op::h_out, tho}})
        .eval();
  }
  return s.status();
}

#if PTENSOR_WITH_QUANTIZED_OPS
int ptensor_quantize_i8(const float* in, int8_t* out, uint32_t count,
                        float scale, int32_t zero_point) {
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_quantize_i8(in + o, out + o, n, scale, zero_point);
    });
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
  if (in && out && count > kMaxDim)
    return chunked(count, 1, [&](uint32_t o, uint32_t n) {
      return ptensor_dequantize_i8(in + o, out + o, n, scale, zero_point);
    });
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

int ptensor_fully_connected_i8_pc(const int8_t* in, float in_scale,
                                  int32_t in_zp, const int8_t* weights,
                                  const float* w_scales, const int32_t* bias,
                                  int8_t* out, float out_scale, int32_t out_zp,
                                  uint32_t batch, uint32_t in_features,
                                  uint32_t out_features, int activation) {
  if (!in || !weights || !w_scales || !out || !fits_dim(batch) ||
      !fits_dim(in_features) || !fits_dim(out_features) ||
      !valid_activation(activation) || !(in_scale > 0) || !(out_scale > 0))
    return invalid();
  for (uint32_t i = 0; i < out_features; ++i)
    if (!(w_scales[i] > 0)) return invalid();
  if (out_features == 1)
    return ptensor_fully_connected_i8(in, in_scale, in_zp, weights,
                                      w_scales[0], bias, out, out_scale,
                                      out_zp, batch, in_features, out_features,
                                      activation);
  std::vector<int32_t> zps(out_features, 0);
  Session s;
  {
    typedef TflmSymQuantOps::FullyConnectedOperator<int8_t> FC;
    Tensor tin = new RomTensor({(uint16_t)batch, (uint16_t)in_features}, i8, in);
    tin->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
    Tensor tw = new RomTensor({(uint16_t)in_features, (uint16_t)out_features}, i8, weights);
    tw->set_quantization_params(
        PerChannelQuantizationParams(zps.data(), w_scales, (int)out_features));
    Tensor tout = new BufferTensor({(uint16_t)batch, (uint16_t)out_features}, i8, out);
    tout->set_quantization_params(PerTensorQuantizationParams(out_zp, out_scale));
    FC op(static_cast<TFLM::TfLiteFusedActivation>(activation));
    if (bias) {
      Tensor tb = new RomTensor({(uint16_t)out_features}, i32, bias);
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

int ptensor_conv2d_i8(const int8_t* in, uint32_t n, uint32_t h, uint32_t w,
                      uint32_t c, float in_scale, int32_t in_zp,
                      const int8_t* filter, uint32_t out_c, uint32_t kh,
                      uint32_t kw, const float* filter_scales,
                      uint32_t n_filter_scales, const int32_t* bias,
                      uint32_t stride_h, uint32_t stride_w, int padding,
                      int activation, int8_t* out, float out_scale,
                      int32_t out_zp) {
  if (!in || !filter || !filter_scales || !out || !fits_dim(n) ||
      !fits_dim(h) || !fits_dim(w) || !fits_dim(c) || !fits_dim(out_c) ||
      !fits_dim(kh) || !fits_dim(kw) || !fits_dim(stride_h) ||
      !fits_dim(stride_w) || !valid_padding(padding) ||
      !valid_activation(activation) || !(in_scale > 0) || !(out_scale > 0) ||
      (n_filter_scales != 1 && n_filter_scales != out_c))
    return invalid();
  for (uint32_t i = 0; i < n_filter_scales; ++i)
    if (!(filter_scales[i] > 0)) return invalid();
  const uint32_t oh = ptensor_conv_output_size(h, kh, stride_h, padding);
  const uint32_t ow = ptensor_conv_output_size(w, kw, stride_w, padding);
  if (!oh || !ow) return invalid();
  // The TFLM kernel always reads a bias; use zeros when there is none.
  std::vector<int32_t> zero_bias;
  if (!bias) {
    zero_bias.assign(out_c, 0);
    bias = zero_bias.data();
  }
  std::vector<int32_t> zps(n_filter_scales, 0);
  Session s;
  {
    typedef TflmSymQuantOps::Conv2dOperator<int8_t> Op;
    Tensor tin = new RomTensor({(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c}, i8, in);
    tin->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
    Tensor tf = new RomTensor({(uint16_t)out_c, (uint16_t)kh, (uint16_t)kw, (uint16_t)c}, i8, filter);
    if (n_filter_scales == 1)
      tf->set_quantization_params(PerTensorQuantizationParams(0, filter_scales[0]));
    else
      tf->set_quantization_params(PerChannelQuantizationParams(
          zps.data(), filter_scales, (int)n_filter_scales));
    Tensor tb = new RomTensor({(uint16_t)out_c}, i32, bias);
    Tensor tout = new BufferTensor({(uint16_t)n, (uint16_t)oh, (uint16_t)ow, (uint16_t)out_c}, i8, out);
    tout->set_quantization_params(PerTensorQuantizationParams(out_zp, out_scale));
    const uint16_t strides[2] = {(uint16_t)stride_h, (uint16_t)stride_w};
    const uint16_t dilation[2] = {1, 1};
    Op op(strides, static_cast<Padding>(padding), dilation,
          static_cast<TFLM::TfLiteFusedActivation>(activation));
    op.set_inputs({{Op::in, tin}, {Op::filter, tf}, {Op::bias, tb}})
        .set_outputs({{Op::out, tout}})
        .eval();
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
int ptensor_fully_connected_i8_pc(const int8_t*, float, int32_t,
                                  const int8_t*, const float*, const int32_t*,
                                  int8_t*, float, int32_t, uint32_t, uint32_t,
                                  uint32_t, int) {
  return invalid();
}
int ptensor_conv2d_i8(const int8_t*, uint32_t, uint32_t, uint32_t, uint32_t,
                      float, int32_t, const int8_t*, uint32_t, uint32_t,
                      uint32_t, const float*, uint32_t, const int32_t*,
                      uint32_t, uint32_t, int, int, int8_t*, float, int32_t) {
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
