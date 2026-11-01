#ifndef PTENSOR_FUSED_OPS_H
#define PTENSOR_FUSED_OPS_H
/**
 * Fused float operators: one pass computes a layer, its (folded) batch norm
 * and its activation.
 *
 *   FusedConv2dOperator<float>           conv2d     -> [batch norm] -> act
 *   FusedDepthwiseConv2dOperator<float>  depthwise  -> [batch norm] -> act
 *   FusedFullyConnectedOperator<float>   dense      -> [batch norm] -> act
 *   FusedAddOperator<float>              add (broadcasting)         -> act
 *
 * Inputs of the conv / dense operators: in, filter, bias (optional) and scale
 * (optional), and per output channel c they compute
 *
 *   out = act(acc * scale[c] + bias[c])   when scale is set
 *   out = act(acc + bias[c])              otherwise
 *
 * A batch norm after the layer folds into these per-channel vectors on the
 * host or at start-up (util/fusion.hpp, python ptensor.fusion). Folding it
 * into the weights too (fold_*_weights) leaves only bias, so the operator
 * costs exactly what the plain layer costs. When the weights cannot be
 * rewritten (they sit in flash as a PackedRomTensor, or are shared), pass
 * the folded scale instead.
 *
 * Without scale, results are bit-identical to running the plain operator and
 * then a ReLU / ReLU6 / clip, because the sums are added in the same order.
 *
 * Every operator takes a dense pointer path when all of its tensors are
 * plain arrays and falls back to per-element tensor reads otherwise (packed
 * or custom tensors), with the same arithmetic.
 *
 * Header-only templates: nothing is emitted unless you instantiate one. The
 * whole file is left out of pTensor.h when PTENSOR_WITH_FUSION is off.
 */
#include <type_traits>

#include "pTensor/core/context.hpp"
#include "pTensor/core/fast_access.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/tensor.hpp"
#include "pTensor/ops/fast/fused_kernels.hpp"
#include "pTensor/ops/reference/ElementwiseOperators.hpp"
#include "pTensor/ops/reference/LayerOperators.hpp"

namespace pTensor {

namespace fused_detail {

// Dense pointer of an optional input, or nullptr.
template <typename Inputs>
inline const float* opt_read(Inputs& inputs, uint8_t name, bool& dense) {
  if (!inputs.has(name)) return nullptr;
  const float* p = fast_read<float>(inputs[name].tensor());
  if (!p) dense = false;
  return p;
}

template <typename Inputs>
inline fused::AnyIn<float> opt_any(Inputs& inputs, uint8_t name) {
  if (!inputs.has(name)) return fused::AnyIn<float>();
  const Tensor& t = inputs[name].tensor();
  return fused::AnyIn<float>(fast_read<float>(t), &t);
}

inline fused::AnyIn<float> any_in(const Tensor& t) {
  return fused::AnyIn<float>(fast_read<float>(t), &t);
}

// The per-channel vectors must hold one value per output channel.
template <typename Inputs>
inline bool channel_vec_ok(Inputs& inputs, uint8_t name, uint32_t channels) {
  return !inputs.has(name) ||
         inputs[name].tensor()->get_shape().get_linear_size() == channels;
}

}  // namespace fused_detail

namespace ReferenceOperators {

/**
 * Float NHWC conv2d with an OHWI filter [out_c, kh, kw, in_c] (the layout of
 * Conv2dOperator<float>), plus folded batch norm and activation.
 */
template <typename T>
class FusedConv2dOperator : public OperatorInterface<4, 1> {
  static_assert(std::is_same<T, float>::value,
                "FusedConv2dOperator is float only; int8 conv already fuses "
                "activation and requantisation (TflmSymQuantOps::Conv2dOperator)");

 public:
  enum names_in : uint8_t { in, filter, bias, scale };
  enum names_out : uint8_t { out };

  FusedConv2dOperator(uint16_t stride_h, uint16_t stride_w, Padding padding,
                      FusedActivation activation = FusedActivation::none(),
                      uint16_t dilation_h = 1, uint16_t dilation_w = 1)
      : _sh(stride_h), _sw(stride_w), _dh(dilation_h), _dw(dilation_w),
        _padding(padding), _act(activation) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    const Tensor& f = inputs[filter].tensor();
    Tensor& y = outputs[out].tensor();
    const TensorShape& is = x->get_shape();
    const TensorShape& fs = f->get_shape();
    const TensorShape& os = y->get_shape();
    if (is.num_dims() != 4 || fs.num_dims() != 4 || os.num_dims() != 4 ||
        fs[3] != is[3] || os[3] != fs[0] || os[0] != is[0] ||
        !fused_detail::channel_vec_ok(inputs, bias, fs[0]) ||
        !fused_detail::channel_vec_ok(inputs, scale, fs[0])) {
      layer_detail::raise_shape_error();
      return;
    }
    fast::ConvGeometry g;
    g.batches = is[0]; g.in_h = is[1]; g.in_w = is[2]; g.in_c = is[3];
    g.out_h = os[1]; g.out_w = os[2]; g.out_c = fs[0];
    g.k_h = fs[1]; g.k_w = fs[2];
    g.stride_h = _sh; g.stride_w = _sw;
    g.dil_h = _dh; g.dil_w = _dw;
    g.pad_top = layer_detail::pad_before(is[1], os[1], fs[1], _sh, _dh, _padding);
    g.pad_left = layer_detail::pad_before(is[2], os[2], fs[2], _sw, _dw, _padding);

    bool dense = true;
    const float* px = fast_read<float>(x);
    const float* pf = fast_read<float>(f);
    float* py = fast_write<float>(y);
    const float* pb = fused_detail::opt_read(inputs, bias, dense);
    const float* ps = fused_detail::opt_read(inputs, scale, dense);
    if (dense && px && pf && py) {
      fused::conv2d_dense(px, pf, pb, ps, py, g, _act);
      return;
    }
    fused::conv2d(fused_detail::any_in(x), fused_detail::any_in(f),
                  fused_detail::opt_any(inputs, bias),
                  fused_detail::opt_any(inputs, scale),
                  fused::AnyOut<float>(py, &y), g, _act);
  }

 private:
  uint16_t _sh, _sw, _dh, _dw;
  Padding _padding;
  FusedActivation _act;
};

/**
 * Float depthwise conv2d, filter [1, kh, kw, in_c * multiplier] (the layout
 * of DepthwiseConv2dOperator<float>), plus folded batch norm and activation.
 */
template <typename T>
class FusedDepthwiseConv2dOperator : public OperatorInterface<4, 1> {
  static_assert(std::is_same<T, float>::value,
                "FusedDepthwiseConv2dOperator is float only");

 public:
  enum names_in : uint8_t { in, filter, bias, scale };
  enum names_out : uint8_t { out };

  FusedDepthwiseConv2dOperator(uint16_t stride_h, uint16_t stride_w,
                               Padding padding,
                               FusedActivation activation = FusedActivation::none(),
                               uint16_t depth_multiplier = 1,
                               uint16_t dilation_h = 1, uint16_t dilation_w = 1)
      : _sh(stride_h), _sw(stride_w), _mult(depth_multiplier),
        _dh(dilation_h), _dw(dilation_w), _padding(padding), _act(activation) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[in].tensor();
    const Tensor& f = inputs[filter].tensor();
    Tensor& y = outputs[out].tensor();
    const TensorShape& is = x->get_shape();
    const TensorShape& fs = f->get_shape();
    const TensorShape& os = y->get_shape();
    if (is.num_dims() != 4 || fs.num_dims() != 4 || os.num_dims() != 4 ||
        _mult == 0 || fs[3] != is[3] * _mult || os[3] != fs[3] ||
        os[0] != is[0] || !fused_detail::channel_vec_ok(inputs, bias, fs[3]) ||
        !fused_detail::channel_vec_ok(inputs, scale, fs[3])) {
      layer_detail::raise_shape_error();
      return;
    }
    fast::ConvGeometry g;
    g.batches = is[0]; g.in_h = is[1]; g.in_w = is[2]; g.in_c = is[3];
    g.out_h = os[1]; g.out_w = os[2]; g.out_c = os[3];
    g.k_h = fs[1]; g.k_w = fs[2];
    g.stride_h = _sh; g.stride_w = _sw;
    g.dil_h = _dh; g.dil_w = _dw;
    g.pad_top = layer_detail::pad_before(is[1], os[1], fs[1], _sh, _dh, _padding);
    g.pad_left = layer_detail::pad_before(is[2], os[2], fs[2], _sw, _dw, _padding);

    bool dense = true;
    const float* px = fast_read<float>(x);
    const float* pf = fast_read<float>(f);
    float* py = fast_write<float>(y);
    const float* pb = fused_detail::opt_read(inputs, bias, dense);
    const float* ps = fused_detail::opt_read(inputs, scale, dense);
    if (dense && px && pf && py) {
      fused::depthwise_conv2d_dense(px, pf, pb, ps, py, g, _mult, _act);
      return;
    }
    fused::depthwise_conv2d(fused_detail::any_in(x), fused_detail::any_in(f),
                            fused_detail::opt_any(inputs, bias),
                            fused_detail::opt_any(inputs, scale),
                            fused::AnyOut<float>(py, &y), g, _mult, _act);
  }

 private:
  uint16_t _sh, _sw, _mult, _dh, _dw;
  Padding _padding;
  FusedActivation _act;
};

/**
 * Float fully connected: out[b, :] = act(in[b, :] @ filter * scale + bias)
 * with filter [in_features, out_features] (the layout of
 * FullyConnectedOperator<float>). The input may have any shape whose size is
 * batch * in_features, so a conv feature map needs no reshape.
 */
template <typename T>
class FusedFullyConnectedOperator : public OperatorInterface<4, 1> {
  static_assert(std::is_same<T, float>::value,
                "FusedFullyConnectedOperator is float only; int8 FC already "
                "fuses activation and requantisation");

 public:
  enum names_in : uint8_t { input, filter, bias, scale };
  enum names_out : uint8_t { output };

  explicit FusedFullyConnectedOperator(
      FusedActivation activation = FusedActivation::none())
      : _act(activation) {}

 protected:
  virtual void compute() {
    const Tensor& x = inputs[input].tensor();
    const Tensor& w = inputs[filter].tensor();
    Tensor& y = outputs[output].tensor();
    const TensorShape& ws = w->get_shape();
    const TensorShape& os = y->get_shape();
    if (ws.num_dims() != 2 || os.num_dims() != 2 || os[1] != ws[1] ||
        x->get_shape().get_linear_size() !=
            static_cast<uint32_t>(os[0]) * ws[0] ||
        !fused_detail::channel_vec_ok(inputs, bias, ws[1]) ||
        !fused_detail::channel_vec_ok(inputs, scale, ws[1])) {
      layer_detail::raise_shape_error();
      return;
    }
    const uint32_t batches = os[0], depth = ws[0], units = ws[1];
    bool dense = true;
    const float* px = fast_read<float>(x);
    const float* pw = fast_read<float>(w);
    float* py = fast_write<float>(y);
    const float* pb = fused_detail::opt_read(inputs, bias, dense);
    const float* ps = fused_detail::opt_read(inputs, scale, dense);
    if (dense && px && pw && py) {
      fused::fully_connected_dense(px, pw, pb, ps, py, batches, depth, units, _act);
      return;
    }
    fused::fully_connected(fused_detail::any_in(x), fused_detail::any_in(w),
                           fused_detail::opt_any(inputs, bias),
                           fused_detail::opt_any(inputs, scale),
                           fused::AnyOut<float>(py, &y), batches, depth, units,
                           _act);
  }

 private:
  FusedActivation _act;
};

/**
 * c = act(a + b) with NumPy broadcasting (e.g. a residual add followed by a
 * ReLU). The output must already have the broadcast shape.
 */
template <typename T>
class FusedAddOperator : public OperatorInterface<2, 1> {
  static_assert(std::is_same<T, float>::value,
                "FusedAddOperator<float>; for int8 use "
                "TflmSymQuantOps::QuantizedAddOperator<int8_t>");

 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };

  explicit FusedAddOperator(FusedActivation activation = FusedActivation::none())
      : _act(activation) {}

 protected:
  virtual void compute() {
    const Tensor& ta = inputs[a].tensor();
    const Tensor& tb = inputs[b].tensor();
    Tensor& tc = outputs[c].tensor();
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
    const fused::AddAct fn(_act);
    const float* pa = fast_read<float>(ta);
    const float* pb = fast_read<float>(tb);
    float* pc = fast_write<float>(tc);
    if (pa && pb && pc) {
      bool same = true;
      for (int i = 0; i < 4; ++i) same = same && va.dims[i] == od[i] && vb.dims[i] == od[i];
      if (same)  // no broadcasting: one flat loop (vectorises)
        fused::add_dense(pa, pb, pc, od[0] * od[1] * od[2] * od[3], _act);
      else
        fast::broadcast_zip(pa, va.strides, pb, vb.strides, pc, od, fn);
      return;
    }
    const fused::AnyIn<float> xa = fused_detail::any_in(ta);
    const fused::AnyIn<float> xb = fused_detail::any_in(tb);
    const fused::AnyOut<float> xc(pc, &tc);
    uint32_t oi = 0;
    for (uint32_t i0 = 0; i0 < od[0]; ++i0)
      for (uint32_t i1 = 0; i1 < od[1]; ++i1)
        for (uint32_t i2 = 0; i2 < od[2]; ++i2) {
          const uint32_t base_a =
              i0 * va.strides[0] + i1 * va.strides[1] + i2 * va.strides[2];
          const uint32_t base_b =
              i0 * vb.strides[0] + i1 * vb.strides[1] + i2 * vb.strides[2];
          for (uint32_t i3 = 0; i3 < od[3]; ++i3, ++oi)
            xc.set(oi, fn(xa[base_a + i3 * va.strides[3]],
                          xb[base_b + i3 * vb.strides[3]]));
        }
  }

 private:
  FusedActivation _act;
};

}  // namespace ReferenceOperators
}  // namespace pTensor
#endif
