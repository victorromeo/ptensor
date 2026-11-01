"""Operator fusion: fold batch norm and activations into the layer before.

Two halves:

* Folding (NumPy only). ``fold_batch_norm`` turns an inference batch norm
  into a per-channel ``scale`` and ``bias``; ``fold_conv2d`` /
  ``fold_depthwise_conv2d`` / ``fold_dense`` also multiply the scale into the
  weights, so the batch norm disappears.

* Fused kernels (the C++ operators in ops/fused/). Each computes a layer, its
  folded batch norm and a clamp activation in one pass::

      y = ptensor.fusion.fused_conv2d(x, w, b, scale=s, activation="relu6")

``fuse(layers)`` applies both to a simple sequential layer list (the format
``from_keras`` produces), and ``run(layers, x)`` executes a list with pTensor's
kernels, so a model can be run fused and unfused and the outputs compared.
"""
import ctypes

import numpy as np

from ._lib import lib
from .ops import (_arr, _check, _f32, _opt_ptr, _padding, _pair, _ptr,
                  _u32arr, PTensorError)
from . import ops as _ops

INF = float("inf")

# Clamp ranges of the activations a fused operator can absorb.
ACTIVATIONS = {
    None: (-INF, INF), "none": (-INF, INF), "linear": (-INF, INF),
    "relu": (0.0, INF), "relu6": (0.0, 6.0), "relu1": (-1.0, 1.0),
}
_ACT_I8 = {None: 0, "none": 0, "linear": 0, "relu": 1, "relu1": 2, "relu6": 3}


def available():
    """True if the loaded library was built with PTENSOR_WITH_FUSION."""
    return bool(lib.ptensor_has_fusion())


def act_range(activation):
    """(lo, hi) clamp for an activation name, a (lo, hi) pair or None."""
    if isinstance(activation, (tuple, list)):
        lo, hi = float(activation[0]), float(activation[1])
        if lo > hi:
            raise ValueError("clamp range must have lo <= hi")
        return lo, hi
    key = activation.lower() if isinstance(activation, str) else activation
    if key not in ACTIVATIONS:
        raise ValueError("activation must be None, 'relu', 'relu6', 'relu1' "
                         "or a (lo, hi) pair, not %r" % (activation,))
    return ACTIVATIONS[key]


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

def fold_batch_norm(mean, variance, gamma=None, beta=None, epsilon=1e-3,
                    bias=None):
    """Per-channel (scale, bias) equal to batch_norm(layer + bias).

    scale = gamma / sqrt(variance + epsilon)
    bias' = (bias - mean) * scale + beta

    Computed in float32 exactly as util/fusion.hpp does on the device.
    """
    mean = np.asarray(mean, np.float32).reshape(-1)
    var = np.asarray(variance, np.float32).reshape(-1)
    c = mean.shape[0]
    g = np.ones(c, np.float32) if gamma is None else np.asarray(gamma, np.float32).reshape(-1)
    bt = np.zeros(c, np.float32) if beta is None else np.asarray(beta, np.float32).reshape(-1)
    b = np.zeros(c, np.float32) if bias is None else np.asarray(bias, np.float32).reshape(-1)
    if not (var.shape[0] == g.shape[0] == bt.shape[0] == b.shape[0] == c):
        raise ValueError("batch norm vectors and bias must have one value per channel")
    scale = (g / np.sqrt(var + np.float32(epsilon))).astype(np.float32)
    new_bias = ((b - mean) * scale + bt).astype(np.float32)
    return scale, new_bias


def _bn(bn):
    return (bn["mean"], bn["variance"], bn.get("gamma"), bn.get("beta"),
            bn.get("epsilon", 1e-3))


def fold_conv2d(filters, bias, bn):
    """Fold batch norm ``bn`` (dict: mean, variance, gamma, beta, epsilon)
    into an OHWI conv filter. Returns (filters', bias')."""
    f = np.asarray(filters, np.float32)
    scale, b = fold_batch_norm(*_bn(bn), bias=bias)
    return (f * scale[:, None, None, None]).astype(np.float32), b


def fold_depthwise_conv2d(filters, bias, bn):
    """Fold batch norm into a [1, kh, kw, c*m] depthwise filter."""
    f = np.asarray(filters, np.float32)
    scale, b = fold_batch_norm(*_bn(bn), bias=bias)
    return (f * scale).astype(np.float32), b


def fold_dense(weights, bias, bn):
    """Fold batch norm into [in, out] dense weights."""
    w = np.asarray(weights, np.float32)
    scale, b = fold_batch_norm(*_bn(bn), bias=bias)
    return (w * scale[None, :]).astype(np.float32), b


# ---------------------------------------------------------------------------
# Fused kernels
# ---------------------------------------------------------------------------

def _vec(v, n, name):
    if v is None:
        return None
    v = _arr(v, np.float32).reshape(-1)
    if v.shape[0] != n:
        raise ValueError("%s must have %d elements" % (name, n))
    return v


def fused_conv2d(x, filters, bias=None, scale=None, stride=1, padding="SAME",
                 activation=None, dilation=1):
    """act(conv2d(x, filters) * scale + bias). x NHWC, filters OHWI."""
    x = _arr(x, np.float32)
    f = _arr(filters, np.float32)
    if x.ndim != 4 or f.ndim != 4 or x.shape[3] != f.shape[3]:
        raise ValueError("fused_conv2d expects x[n,h,w,c] and filters[out_c,kh,kw,c]")
    n, h, w, c = x.shape
    oc, kh, kw, _ = f.shape
    b, s = _vec(bias, oc, "bias"), _vec(scale, oc, "scale")
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    pad = _padding(padding)
    lo, hi = act_range(activation)
    oh = lib.ptensor_conv_output_size(h, (kh - 1) * dh + 1, sh, pad)
    ow = lib.ptensor_conv_output_size(w, (kw - 1) * dw + 1, sw, pad)
    if not oh or not ow:
        raise ValueError("filter does not fit the input")
    out = np.empty((n, oh, ow, oc), np.float32)
    _check("fused_conv2d", lib.ptensor_fused_conv2d_f32(
        _f32(x), n, h, w, c, _f32(f), oc, kh, kw, _opt_ptr(b, ctypes.c_float),
        _opt_ptr(s, ctypes.c_float), sh, sw, pad, dh, dw, lo, hi, _f32(out)))
    return out


def fused_depthwise_conv2d(x, filters, bias=None, scale=None, stride=1,
                           padding="SAME", activation=None, depth_multiplier=1,
                           dilation=1):
    """act(depthwise_conv2d(x, filters) * scale + bias), filters [1,kh,kw,c*m]."""
    x = _arr(x, np.float32)
    f = _arr(filters, np.float32)
    if x.ndim != 4 or f.ndim != 4 or f.shape[0] != 1:
        raise ValueError("fused_depthwise_conv2d expects x[n,h,w,c] and filters[1,kh,kw,c*m]")
    n, h, w, c = x.shape
    _, kh, kw, oc = f.shape
    m = int(depth_multiplier)
    if oc != c * m:
        raise ValueError("filters.shape[3] must be c * depth_multiplier")
    b, s = _vec(bias, oc, "bias"), _vec(scale, oc, "scale")
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    pad = _padding(padding)
    lo, hi = act_range(activation)
    oh = lib.ptensor_conv_output_size(h, (kh - 1) * dh + 1, sh, pad)
    ow = lib.ptensor_conv_output_size(w, (kw - 1) * dw + 1, sw, pad)
    if not oh or not ow:
        raise ValueError("kernel does not fit the input")
    out = np.empty((n, oh, ow, oc), np.float32)
    _check("fused_depthwise_conv2d", lib.ptensor_fused_depthwise_conv2d_f32(
        _f32(x), n, h, w, c, _f32(f), kh, kw, m, _opt_ptr(b, ctypes.c_float),
        _opt_ptr(s, ctypes.c_float), sh, sw, pad, dh, dw, lo, hi, _f32(out)))
    return out


def fused_fully_connected(x, weights, bias=None, scale=None, activation=None):
    """act(x[batch,in] @ weights[in,out] * scale + bias)."""
    x = _arr(np.atleast_2d(x), np.float32)
    w = _arr(weights, np.float32)
    if w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("fused_fully_connected expects x[batch,in] and weights[in,out]")
    b, s = _vec(bias, w.shape[1], "bias"), _vec(scale, w.shape[1], "scale")
    lo, hi = act_range(activation)
    out = np.empty((x.shape[0], w.shape[1]), np.float32)
    _check("fused_fully_connected", lib.ptensor_fused_fully_connected_f32(
        _f32(x), _f32(w), _opt_ptr(b, ctypes.c_float), _opt_ptr(s, ctypes.c_float),
        _f32(out), x.shape[0], x.shape[1], w.shape[1], lo, hi))
    return out


def fused_add(a, b, activation=None):
    """act(a + b) with NumPy broadcasting (rank <= 4)."""
    a = _arr(a, np.float32)
    b = _arr(b, np.float32)
    if max(a.ndim, b.ndim) > 4:
        raise ValueError("fused_add supports rank <= 4")
    lo, hi = act_range(activation)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), np.float32)
    sa, pa = _u32arr(a.shape)
    sb, pb = _u32arr(b.shape)
    _check("fused_add", lib.ptensor_fused_add_f32(
        _f32(a), pa, max(a.ndim, 1), _f32(b), pb, max(b.ndim, 1), lo, hi, _f32(out)))
    return out


def add_int8(a, a_scale, a_zero_point, b, b_scale, b_zero_point, out_scale,
             out_zero_point, activation=None):
    """int8 add with fused activation and requantisation (TFLite ADD)."""
    a = _arr(a, np.int8)
    b = _arr(b, np.int8)
    if max(a.ndim, b.ndim) > 4:
        raise ValueError("add_int8 supports rank <= 4")
    key = activation.lower() if isinstance(activation, str) else activation
    if key not in _ACT_I8:
        raise ValueError("activation must be None, 'relu', 'relu1' or 'relu6'")
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), np.int8)
    sa, pa = _u32arr(a.shape)
    sb, pb = _u32arr(b.shape)
    _check("add_int8", lib.ptensor_add_i8(
        _ptr(a, ctypes.c_int8), pa, max(a.ndim, 1), a_scale, a_zero_point,
        _ptr(b, ctypes.c_int8), pb, max(b.ndim, 1), b_scale, b_zero_point,
        _ptr(out, ctypes.c_int8), out_scale, out_zero_point, _ACT_I8[key]))
    return out


# ---------------------------------------------------------------------------
# Sequential layer lists
# ---------------------------------------------------------------------------
#
# A layer is a dict with an "op" key:
#   conv2d:            filters (OHWI), bias, stride, padding, activation
#   depthwise_conv2d:  filters ([1,kh,kw,c*m]), bias, stride, padding,
#                      depth_multiplier, activation
#   dense:             weights ([in,out]), bias, activation  (flattens NHWC input)
#   batch_norm:        mean, variance, gamma, beta, epsilon
#   activation:        activation ("relu", "relu6", "relu1" or (lo, hi))
#   maxpool2d / avgpool2d: kernel, stride, padding
#   flatten
# Layers may also carry "scale" (per-channel, from fuse(fold_weights=False)).

_LINEAR = ("conv2d", "depthwise_conv2d", "dense")


def _compose(first, second):
    """Clamp range of applying activation `first` then `second`."""
    lo1, hi1 = act_range(first)
    lo2, hi2 = act_range(second)
    lo, hi = max(lo1, lo2), min(hi1, hi2)
    if lo > hi:  # disjoint: everything ends up at one bound of `second`
        lo = hi = lo2 if hi1 < lo2 else hi2
    return lo, hi


def fuse(layers, fold_weights=True):
    """Fuse a sequential layer list: every conv2d / depthwise_conv2d / dense
    absorbs the batch_norm and activation layers that directly follow it.

    fold_weights=True multiplies the batch-norm scale into the weights (zero
    run-time cost). False keeps the weights as they are and stores the scale
    in the layer instead (use when the weights are stored packed / encoded).

    Returns (fused_layers, report) where report lists what was folded.
    """
    out, report = [], []
    i = 0
    while i < len(layers):
        layer = dict(layers[i])
        i += 1
        if layer["op"] not in _LINEAR:
            out.append(layer)
            continue
        absorbed = []
        act = layer.get("activation")
        while i < len(layers):
            nxt = layers[i]
            if nxt["op"] == "batch_norm" and act_range(act) == act_range(None) \
                    and "scale" not in layer:
                key = "filters" if "filters" in layer else "weights"
                if fold_weights:
                    fold = {"conv2d": fold_conv2d, "depthwise_conv2d": fold_depthwise_conv2d,
                            "dense": fold_dense}[layer["op"]]
                    layer[key], layer["bias"] = fold(layer[key], layer.get("bias"), nxt)
                else:
                    layer["scale"], layer["bias"] = fold_batch_norm(*_bn(nxt), bias=layer.get("bias"))
                absorbed.append("batch_norm")
            elif nxt["op"] == "activation":
                act = _compose(act, nxt["activation"])
                absorbed.append(str(nxt["activation"]))
            else:
                break
            i += 1
        if act is not None:
            layer["activation"] = act
        out.append(layer)
        report.append((layer["op"], absorbed))
    return out, report


def run(layers, x):
    """Run a layer list on float input x (NHWC, or [batch, features]) with
    pTensor's kernels: fused operators for conv / depthwise / dense layers,
    the standalone operators for batch_norm and activation layers."""
    y = np.asarray(x, np.float32)
    for layer in layers:
        op = layer["op"]
        act = layer.get("activation")
        if op == "conv2d":
            y = fused_conv2d(y, layer["filters"], layer.get("bias"), layer.get("scale"),
                             layer.get("stride", 1), layer.get("padding", "SAME"), act)
        elif op == "depthwise_conv2d":
            y = fused_depthwise_conv2d(y, layer["filters"], layer.get("bias"),
                                       layer.get("scale"), layer.get("stride", 1),
                                       layer.get("padding", "SAME"), act,
                                       layer.get("depth_multiplier", 1))
        elif op == "dense":
            y = fused_fully_connected(y.reshape(y.shape[0], -1), layer["weights"],
                                      layer.get("bias"), layer.get("scale"), act)
        elif op == "batch_norm":
            y = _ops.batch_norm(y, layer["mean"], layer["variance"],
                                layer.get("gamma", np.ones_like(layer["mean"])),
                                layer.get("beta", np.zeros_like(layer["mean"])),
                                layer.get("epsilon", 1e-3))
        elif op == "activation":
            lo, hi = act_range(layer["activation"])
            y = np.ascontiguousarray(y)
            if (lo, hi) == (0.0, INF):
                y = _ops.relu(y)
            elif (lo, hi) == (0.0, 6.0):
                y = _ops.relu6(y)
            else:
                y = _ops.clip(y, lo, hi)
        elif op in ("maxpool2d", "avgpool2d"):
            fn = _ops.maxpool2d if op == "maxpool2d" else _ops.avgpool2d
            y = fn(y, layer["kernel"], layer.get("stride"), layer.get("padding", "VALID"))
        elif op == "flatten":
            y = y.reshape(y.shape[0], -1)
        else:
            raise ValueError("unknown layer op %r" % (op,))
    return y


def from_keras(model):
    """Layer list for a Keras Sequential model built from Conv2D,
    DepthwiseConv2D, Dense, BatchNormalization, ReLU / Activation,
    MaxPooling2D, AveragePooling2D, Flatten, Dropout and InputLayer."""
    layers = []
    for l in model.layers:
        cls = type(l).__name__
        cfg = l.get_config()
        act = cfg.get("activation", "linear")
        if cls in ("Conv2D", "DepthwiseConv2D", "Dense") and act not in ACTIVATIONS:
            raise NotImplementedError("activation %s in %s" % (act, l.name))
        act = None if act == "linear" else act
        bias = l.bias.numpy() if getattr(l, "use_bias", False) else None
        if cls == "Conv2D":
            if tuple(cfg.get("dilation_rate", (1, 1))) != (1, 1) or cfg.get("groups", 1) != 1:
                raise NotImplementedError("dilated / grouped Conv2D")
            layers.append({"op": "conv2d", "filters": np.transpose(l.kernel.numpy(), (3, 0, 1, 2)),
                           "bias": bias, "stride": tuple(cfg["strides"]),
                           "padding": cfg["padding"].upper(), "activation": act})
        elif cls == "DepthwiseConv2D":
            k = l.kernel.numpy()  # [kh, kw, c, m]
            kh, kw, c, m = k.shape
            layers.append({"op": "depthwise_conv2d", "filters": k.reshape(1, kh, kw, c * m),
                           "bias": bias, "stride": tuple(cfg["strides"]),
                           "padding": cfg["padding"].upper(), "depth_multiplier": m,
                           "activation": act})
        elif cls == "Dense":
            layers.append({"op": "dense", "weights": l.kernel.numpy(), "bias": bias,
                           "activation": act})
        elif cls == "BatchNormalization":
            axis = cfg["axis"]
            axis = axis[0] if isinstance(axis, (list, tuple)) else axis
            if axis not in (-1, len(l.input.shape) - 1):
                raise NotImplementedError("BatchNormalization over a non-channel axis")
            layers.append({"op": "batch_norm", "mean": l.moving_mean.numpy(),
                           "variance": l.moving_variance.numpy(),
                           "gamma": l.gamma.numpy() if l.gamma is not None else None,
                           "beta": l.beta.numpy() if l.beta is not None else None,
                           "epsilon": cfg["epsilon"]})
        elif cls == "ReLU":
            if cfg.get("negative_slope", 0) or cfg.get("threshold", 0):
                raise NotImplementedError("leaky / thresholded ReLU")
            mx = cfg.get("max_value")
            layers.append({"op": "activation",
                           "activation": "relu" if mx is None else (0.0, float(mx))})
        elif cls == "Activation":
            if act not in ACTIVATIONS:
                raise NotImplementedError("activation %s" % act)
            if act is not None:
                layers.append({"op": "activation", "activation": act})
        elif cls in ("MaxPooling2D", "AveragePooling2D"):
            layers.append({"op": "maxpool2d" if cls == "MaxPooling2D" else "avgpool2d",
                           "kernel": tuple(cfg["pool_size"]), "stride": tuple(cfg["strides"]),
                           "padding": cfg["padding"].upper()})
        elif cls == "Flatten":
            layers.append({"op": "flatten"})
        elif cls in ("InputLayer", "Dropout"):
            continue
        else:
            raise NotImplementedError(cls)
    return layers


__all__ = [
    "available", "act_range", "fold_batch_norm", "fold_conv2d",
    "fold_depthwise_conv2d", "fold_dense", "fused_conv2d",
    "fused_depthwise_conv2d", "fused_fully_connected", "fused_add",
    "add_int8", "fuse", "run", "from_keras", "PTensorError",
]
