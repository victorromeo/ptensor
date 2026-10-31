#ifndef PTENSOR_LAYER_OPS_H
#define PTENSOR_LAYER_OPS_H
/**
 * Float reference implementations of common network layers that the original
 * operator set lacked. All are header-only templates (zero cost if unused).
 *
 * Layout conventions follow TFLite: images are NHWC, depthwise filters are
 * [1, kh, kw, channels * multiplier].
 */
#include <cmath>

#include "pTensor/core/context.hpp"
#include "pTensor/core/errorHandler.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"

namespace pTensor {

DECLARE_ERROR(InvalidLayerShapeError);

namespace layer_detail {

inline void raise_shape_error() {
  Context::get_default_context()->throwError(InvalidLayerShapeError());
}

inline uint32_t last_dim(const TensorShape& s) {
  return s[s.num_dims() - 1];
}

// Output-size-consistent "SAME"/"VALID" padding offset, TensorFlow semantics.
inline int32_t pad_before(uint32_t in, uint32_t out, uint32_t k,
                          uint32_t stride, uint32_t dilation, Padding p) {
  if (p != SAME) return 0;
  const int32_t eff_k = static_cast<int32_t>((k - 1) * dilation + 1);
  const int32_t total =
      static_cast<int32_t>((out - 1) * stride) + eff_k - static_cast<int32_t>(in);
  return total > 0 ? total / 2 : 0;
}

}  // namespace layer_detail

namespace ReferenceOperators {

/**
 * Inference-time batch normalisation over the last (channel) axis:
 *   out = (in - mean) / sqrt(variance + epsilon) * gamma + beta
 * Inputs: in, mean, variance, gamma, beta (each of the last four is [C]).
 */
template <typename T>
class BatchNormOperator : public OperatorInterface<5, 1> {
 public:
  enum names_in : uint8_t { in, mean, variance, gamma, beta };
  enum names_out : uint8_t { out };

  explicit BatchNormOperator(float epsilon = 1e-3f) : _epsilon(epsilon) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    Tensor& y = outputs[out].tensor();
    const uint32_t n = x->get_shape().get_linear_size();
    const uint32_t c = layer_detail::last_dim(x->get_shape());
    if (n != y->get_shape().get_linear_size() ||
        inputs[mean].tensor()->get_shape().get_linear_size() != c) {
      layer_detail::raise_shape_error();
      return;
    }
    const Tensor& m = inputs[mean].tensor();
    const Tensor& v = inputs[variance].tensor();
    const Tensor& g = inputs[gamma].tensor();
    const Tensor& b = inputs[beta].tensor();
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t ch = i % c;
      const float scale = static_cast<float>(static_cast<T>(g(ch))) /
                          std::sqrt(static_cast<float>(static_cast<T>(v(ch))) + _epsilon);
      const float xv = static_cast<T>(x(i));
      const float mv = static_cast<T>(m(ch));
      const float bv = static_cast<T>(b(ch));
      y(i) = static_cast<T>((xv - mv) * scale + bv);
    }
  }

 private:
  float _epsilon;
};

/**
 * Per-channel affine transform out = in * scale + shift, broadcasting [C]
 * along the last axis. This is what a folded batch norm becomes.
 */
template <typename T>
class ChannelAffineOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, scale, shift };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    const Tensor& s = inputs[scale].tensor();
    const Tensor& t = inputs[shift].tensor();
    Tensor& y = outputs[out].tensor();
    const uint32_t n = x->get_shape().get_linear_size();
    const uint32_t c = layer_detail::last_dim(x->get_shape());
    if (n != y->get_shape().get_linear_size() ||
        s->get_shape().get_linear_size() != c) {
      layer_detail::raise_shape_error();
      return;
    }
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t ch = i % c;
      const T xv = x(i);
      const T sv = s(ch);
      const T tv = t(ch);
      y(i) = static_cast<T>(xv * sv + tv);
    }
  }
};

/**
 * Constant padding of a 4D NHWC tensor. pads = {top, bottom, left, right}
 * (batch and channels are not padded).
 */
template <typename T>
class PadOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  PadOperator(uint16_t top, uint16_t bottom, uint16_t left, uint16_t right,
              T value = 0)
      : _top(top), _bottom(bottom), _left(left), _right(right), _value(value) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    Tensor& y = outputs[out].tensor();
    const TensorShape& is = x->get_shape();
    const TensorShape& os = y->get_shape();
    if (is.num_dims() != 4 || os.num_dims() != 4 || os[0] != is[0] ||
        os[3] != is[3] || os[1] != is[1] + _top + _bottom ||
        os[2] != is[2] + _left + _right) {
      layer_detail::raise_shape_error();
      return;
    }
    for (uint16_t n = 0; n < os[0]; ++n)
      for (uint16_t h = 0; h < os[1]; ++h)
        for (uint16_t w = 0; w < os[2]; ++w) {
          const int32_t ih = static_cast<int32_t>(h) - _top;
          const int32_t iw = static_cast<int32_t>(w) - _left;
          const bool inside = ih >= 0 && ih < is[1] && iw >= 0 && iw < is[2];
          for (uint16_t c = 0; c < os[3]; ++c) {
            if (inside) {
              const T v = x(n, static_cast<uint16_t>(ih), static_cast<uint16_t>(iw), c);
              y(n, h, w, c) = v;
            } else {
              y(n, h, w, c) = _value;
            }
          }
        }
  }

 private:
  uint16_t _top, _bottom, _left, _right;
  T _value;
};

/** Concatenate two tensors of equal rank along `axis`. */
template <typename T>
class ConcatOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { out };

  explicit ConcatOperator(int8_t axis) : _axis(axis) {}

 protected:
  virtual void compute() {
    const Tensor& ta = inputs[a].tensor();
    const Tensor& tb = inputs[b].tensor();
    Tensor& to = outputs[out].tensor();
    const TensorShape& sa = ta->get_shape();
    const TensorShape& sb = tb->get_shape();
    const TensorShape& so = to->get_shape();
    const int rank = sa.num_dims();
    const int axis = _axis < 0 ? _axis + rank : _axis;
    bool ok = rank == sb.num_dims() && rank == so.num_dims() && axis >= 0 &&
              axis < rank;
    uint32_t outer = 1, inner = 1;
    for (int d = 0; ok && d < rank; ++d) {
      if (d == axis) {
        ok = so[d] == sa[d] + sb[d];
      } else {
        ok = sa[d] == sb[d] && sa[d] == so[d];
        if (d < axis) outer *= sa[d];
        else inner *= sa[d];
      }
    }
    if (!ok) {
      layer_detail::raise_shape_error();
      return;
    }
    const uint32_t ca = sa[axis] * inner, cb = sb[axis] * inner;
    uint32_t oi = 0;
    for (uint32_t o = 0; o < outer; ++o) {
      for (uint32_t i = 0; i < ca; ++i, ++oi) {
        const T v = ta(o * ca + i);
        to(oi) = v;
      }
      for (uint32_t i = 0; i < cb; ++i, ++oi) {
        const T v = tb(o * cb + i);
        to(oi) = v;
      }
    }
  }

 private:
  int8_t _axis;
};

/** Mean over H and W of an NHWC tensor: [N,H,W,C] -> [N,1,1,C] (or [N,C]). */
template <typename T>
class GlobalAveragePoolOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    Tensor& y = outputs[out].tensor();
    const TensorShape& s = x->get_shape();
    if (s.num_dims() != 4 ||
        y->get_shape().get_linear_size() != static_cast<uint32_t>(s[0]) * s[3]) {
      layer_detail::raise_shape_error();
      return;
    }
    const float inv = 1.0f / (static_cast<float>(s[1]) * s[2]);
    for (uint16_t n = 0; n < s[0]; ++n)
      for (uint16_t c = 0; c < s[3]; ++c) {
        float acc = 0.0f;
        for (uint16_t h = 0; h < s[1]; ++h)
          for (uint16_t w = 0; w < s[2]; ++w) {
            acc += static_cast<float>(static_cast<T>(x(n, h, w, c)));
          }
        y(static_cast<uint32_t>(n) * s[3] + c) = static_cast<T>(acc * inv);
      }
  }
};

/**
 * Float depthwise 2D convolution (TFLite DEPTHWISE_CONV_2D).
 * in: [N,H,W,C], filter: [1,KH,KW,C*M], bias: [C*M] (optional),
 * out: [N,OH,OW,C*M].
 */
template <typename T>
class DepthwiseConv2dOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

  DepthwiseConv2dOperator(uint16_t stride_h, uint16_t stride_w,
                          Padding padding, uint16_t depth_multiplier = 1,
                          uint16_t dilation_h = 1, uint16_t dilation_w = 1)
      : _sh(stride_h), _sw(stride_w), _padding(padding),
        _mult(depth_multiplier), _dh(dilation_h), _dw(dilation_w) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    const Tensor& f = inputs[filter].tensor();
    Tensor& y = outputs[out].tensor();
    const bool have_bias = inputs.has(bias);
    const TensorShape& is = x->get_shape();
    const TensorShape& fs = f->get_shape();
    const TensorShape& os = y->get_shape();
    if (is.num_dims() != 4 || fs.num_dims() != 4 || os.num_dims() != 4 ||
        fs[3] != is[3] * _mult || os[3] != fs[3] || os[0] != is[0]) {
      layer_detail::raise_shape_error();
      return;
    }
    const uint16_t kh = fs[1], kw = fs[2];
    const int32_t pt = layer_detail::pad_before(is[1], os[1], kh, _sh, _dh, _padding);
    const int32_t pl = layer_detail::pad_before(is[2], os[2], kw, _sw, _dw, _padding);
    for (uint16_t n = 0; n < os[0]; ++n)
      for (uint16_t oy = 0; oy < os[1]; ++oy)
        for (uint16_t ox = 0; ox < os[2]; ++ox)
          for (uint16_t ic = 0; ic < is[3]; ++ic)
            for (uint16_t m = 0; m < _mult; ++m) {
              const uint16_t oc = static_cast<uint16_t>(ic * _mult + m);
              float acc = have_bias
                              ? static_cast<float>(static_cast<T>(inputs[bias].tensor()(oc)))
                              : 0.0f;
              for (uint16_t ky = 0; ky < kh; ++ky) {
                const int32_t iy = oy * _sh - pt + ky * _dh;
                if (iy < 0 || iy >= is[1]) continue;
                for (uint16_t kx = 0; kx < kw; ++kx) {
                  const int32_t ix = ox * _sw - pl + kx * _dw;
                  if (ix < 0 || ix >= is[2]) continue;
                  const float xv = static_cast<T>(
                      x(n, static_cast<uint16_t>(iy), static_cast<uint16_t>(ix), ic));
                  const float wv = static_cast<T>(f(0, ky, kx, oc));
                  acc += xv * wv;
                }
              }
              y(n, oy, ox, oc) = static_cast<T>(acc);
            }
  }

 private:
  uint16_t _sh, _sw;
  Padding _padding;
  uint16_t _mult, _dh, _dw;
};

namespace rnn_detail {
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace rnn_detail

/**
 * One time step of an LSTM cell (Keras gate order i, f, c, o):
 *   z = x @ W + h @ U + b              W: [in, 4H], U: [H, 4H], b: [4H]
 *   i, f, g, o = sigmoid(z_i), sigmoid(z_f), tanh(z_c), sigmoid(z_o)
 *   c' = f * c + i * g ;  h' = o * tanh(c')
 * Inputs x: [B, in], h: [B, H], c: [B, H]. Outputs h': [B, H], c': [B, H].
 * Run it once per time step; h'/c' may alias the next step's h/c buffers only
 * through separate tensors (in-place update is not supported).
 */
class LstmCellOperator : public OperatorInterface<6, 2> {
 public:
  enum names_in : uint8_t { x, h, c, W, U, b };
  enum names_out : uint8_t { h_out, c_out };

 protected:
  virtual void compute() {
    const Tensor& tx = inputs[x].tensor();
    const Tensor& th = inputs[h].tensor();
    const Tensor& tc = inputs[c].tensor();
    const Tensor& tW = inputs[W].tensor();
    const Tensor& tU = inputs[U].tensor();
    const Tensor& tb = inputs[b].tensor();
    Tensor& ho = outputs[h_out].tensor();
    Tensor& co = outputs[c_out].tensor();
    const TensorShape& xs = tx->get_shape();
    const TensorShape& hs = th->get_shape();
    const uint16_t B = xs[0], I = xs[1], H = hs[1];
    if (xs.num_dims() != 2 || hs.num_dims() != 2 || hs[0] != B ||
        tW->get_shape()[0] != I || tW->get_shape()[1] != 4 * H ||
        tU->get_shape()[0] != H || tU->get_shape()[1] != 4 * H ||
        tb->get_shape().get_linear_size() != 4u * H ||
        ho->get_shape().get_linear_size() != static_cast<uint32_t>(B) * H ||
        co->get_shape().get_linear_size() != static_cast<uint32_t>(B) * H) {
      layer_detail::raise_shape_error();
      return;
    }
    for (uint16_t bi = 0; bi < B; ++bi) {
      for (uint16_t j = 0; j < H; ++j) {
        float z[4];
        for (int g = 0; g < 4; ++g) {
          const uint16_t col = static_cast<uint16_t>(g * H + j);
          float acc = static_cast<float>(tb(col));
          for (uint16_t k = 0; k < I; ++k)
            acc += static_cast<float>(tx(bi, k)) * static_cast<float>(tW(k, col));
          for (uint16_t k = 0; k < H; ++k)
            acc += static_cast<float>(th(bi, k)) * static_cast<float>(tU(k, col));
          z[g] = acc;
        }
        const float ig = rnn_detail::sigmoid(z[0]);
        const float fg = rnn_detail::sigmoid(z[1]);
        const float gg = std::tanh(z[2]);
        const float og = rnn_detail::sigmoid(z[3]);
        const float cn = fg * static_cast<float>(tc(bi, j)) + ig * gg;
        co(bi, j) = cn;
        ho(bi, j) = og * std::tanh(cn);
      }
    }
  }
};

/**
 * One time step of a GRU cell (Keras gate order z, r, h; reset_after=true):
 *   xz, xr, xh = x @ W + bx            W: [in, 3H], bx: [3H]
 *   hz, hr, hh = h @ U + bh            U: [H, 3H],  bh: [3H]
 *   z = sigmoid(xz + hz); r = sigmoid(xr + hr); n = tanh(xh + r * hh)
 *   h' = z * h + (1 - z) * n
 */
class GruCellOperator : public OperatorInterface<6, 1> {
 public:
  enum names_in : uint8_t { x, h, W, U, bx, bh };
  enum names_out : uint8_t { h_out };

 protected:
  virtual void compute() {
    const Tensor& tx = inputs[x].tensor();
    const Tensor& th = inputs[h].tensor();
    const Tensor& tW = inputs[W].tensor();
    const Tensor& tU = inputs[U].tensor();
    const Tensor& tbx = inputs[bx].tensor();
    const Tensor& tbh = inputs[bh].tensor();
    Tensor& ho = outputs[h_out].tensor();
    const TensorShape& xs = tx->get_shape();
    const TensorShape& hs = th->get_shape();
    const uint16_t B = xs[0], I = xs[1], H = hs[1];
    if (xs.num_dims() != 2 || hs.num_dims() != 2 || hs[0] != B ||
        tW->get_shape()[0] != I || tW->get_shape()[1] != 3 * H ||
        tU->get_shape()[0] != H || tU->get_shape()[1] != 3 * H ||
        tbx->get_shape().get_linear_size() != 3u * H ||
        tbh->get_shape().get_linear_size() != 3u * H ||
        ho->get_shape().get_linear_size() != static_cast<uint32_t>(B) * H) {
      layer_detail::raise_shape_error();
      return;
    }
    for (uint16_t bi = 0; bi < B; ++bi) {
      for (uint16_t j = 0; j < H; ++j) {
        float xg[3], hg[3];
        for (int g = 0; g < 3; ++g) {
          const uint16_t col = static_cast<uint16_t>(g * H + j);
          float ax = static_cast<float>(tbx(col));
          float ah = static_cast<float>(tbh(col));
          for (uint16_t k = 0; k < I; ++k)
            ax += static_cast<float>(tx(bi, k)) * static_cast<float>(tW(k, col));
          for (uint16_t k = 0; k < H; ++k)
            ah += static_cast<float>(th(bi, k)) * static_cast<float>(tU(k, col));
          xg[g] = ax;
          hg[g] = ah;
        }
        const float z = rnn_detail::sigmoid(xg[0] + hg[0]);
        const float r = rnn_detail::sigmoid(xg[1] + hg[1]);
        const float n = std::tanh(xg[2] + r * hg[2]);
        const float hprev = static_cast<float>(th(bi, j));
        ho(bi, j) = z * hprev + (1.0f - z) * n;
      }
    }
  }
};

}  // namespace ReferenceOperators
}  // namespace pTensor
#endif
