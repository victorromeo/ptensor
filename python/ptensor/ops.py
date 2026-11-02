"""NumPy-friendly wrappers around the pTensor operators."""
import ctypes

import numpy as np

from ._lib import lib

PADDING = {"VALID": 1, "SAME": 2}
ACTIVATION = {None: 0, "none": 0, "relu": 1, "relu6": 3}


class PTensorError(RuntimeError):
    """Raised when a pTensor call fails."""

    def __init__(self, name, status):
        err = lib.ptensor_last_error()
        kind = {1: "invalid argument", 2: "runtime error"}.get(status, "error")
        msg = "%s failed: %s" % (name, kind)
        if err:
            msg += " (pTensor error id 0x%04x)" % err
        super().__init__(msg)
        self.status = status
        self.error_id = err


def _check(name, status):
    if status != 0:
        raise PTensorError(name, status)


def _arr(x, dtype):
    return np.ascontiguousarray(x, dtype=dtype)


def _ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _opt_ptr(a, ctype):
    return None if a is None else _ptr(a, ctype)


def _f32(a):
    return _ptr(a, ctypes.c_float)


def _padding(p):
    try:
        return PADDING[p.upper()]
    except (KeyError, AttributeError):
        raise ValueError("padding must be 'SAME' or 'VALID'")


def _activation(a):
    key = a.lower() if isinstance(a, str) else a
    if key not in ACTIVATION:
        raise ValueError("activation must be None, 'relu' or 'relu6'")
    return ACTIVATION[key]


def _pair(v):
    if isinstance(v, int):
        return v, v
    a, b = v
    return int(a), int(b)


def conv_output_size(in_size, kernel, stride, padding):
    """Spatial output size of a conv/pool window (TensorFlow semantics)."""
    return int(lib.ptensor_conv_output_size(in_size, kernel, stride, _padding(padding)))


def matmul(a, b):
    """a[m,k] @ b[k,n] in float32."""
    a = _arr(a, np.float32)
    b = _arr(b, np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul expects [m,k] @ [k,n]")
    out = np.empty((a.shape[0], b.shape[1]), np.float32)
    _check("matmul", lib.ptensor_matmul_f32(_f32(a), _f32(b), _f32(out),
                                            a.shape[0], a.shape[1], b.shape[1]))
    return out


def fully_connected(x, weights, bias=None, activation=None):
    """act(x[batch,in] @ weights[in,out] + bias) in float32."""
    x = _arr(np.atleast_2d(x), np.float32)
    w = _arr(weights, np.float32)
    if w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("fully_connected expects x[batch,in] and weights[in,out]")
    b = None if bias is None else _arr(bias, np.float32).reshape(-1)
    if b is not None and b.shape[0] != w.shape[1]:
        raise ValueError("bias must have out_features elements")
    out = np.empty((x.shape[0], w.shape[1]), np.float32)
    _check("fully_connected", lib.ptensor_fully_connected_f32(
        _f32(x), _f32(w), _opt_ptr(b, ctypes.c_float), _f32(out),
        x.shape[0], x.shape[1], w.shape[1], _activation(activation)))
    return out


def add(a, b):
    """Elementwise a + b (same shape) in float32."""
    a = _arr(a, np.float32)
    b = _arr(b, np.float32)
    if a.shape != b.shape:
        raise ValueError("add expects equal shapes")
    out = np.empty_like(a)
    _check("add", lib.ptensor_add_f32(_f32(a), _f32(b), _f32(out), a.size))
    return out


def _unary(name, fn):
    def op(x):
        x = _arr(x, np.float32)
        out = np.empty_like(x)
        _check(name, fn(_f32(x), _f32(out), x.size))
        return out
    op.__name__ = name
    op.__doc__ = "Elementwise %s in float32." % name
    return op


relu = _unary("relu", lib.ptensor_relu_f32)
relu6 = _unary("relu6", lib.ptensor_relu6_f32)
sigmoid = _unary("sigmoid", lib.ptensor_sigmoid_f32)
tanh = _unary("tanh", lib.ptensor_tanh_f32)


def softmax(x):
    """Softmax over the last axis in float32."""
    x = _arr(x, np.float32)
    rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    out = np.empty_like(x)
    _check("softmax", lib.ptensor_softmax_f32(_f32(x), _f32(out), rows, x.shape[-1]))
    return out


def conv2d(x, filters, bias=None, stride=1, padding="SAME"):
    """2D convolution, x: NHWC, filters: [out_c, kh, kw, in_c] (OHWI)."""
    x = _arr(x, np.float32)
    f = _arr(filters, np.float32)
    if x.ndim != 4 or f.ndim != 4 or x.shape[3] != f.shape[3]:
        raise ValueError("conv2d expects x[n,h,w,c] and filters[out_c,kh,kw,c]")
    sh, sw = _pair(stride)
    pad = _padding(padding)
    n, h, w, c = x.shape
    oc, kh, kw, _ = f.shape
    oh = lib.ptensor_conv_output_size(h, kh, sh, pad)
    ow = lib.ptensor_conv_output_size(w, kw, sw, pad)
    if not oh or not ow:
        raise ValueError("kernel does not fit the input")
    b = None if bias is None else _arr(bias, np.float32).reshape(-1)
    if b is not None and b.shape[0] != oc:
        raise ValueError("bias must have out_c elements")
    out = np.empty((n, oh, ow, oc), np.float32)
    _check("conv2d", lib.ptensor_conv2d_f32(
        _f32(x), n, h, w, c, _f32(f), oc, kh, kw, _opt_ptr(b, ctypes.c_float),
        sh, sw, pad, _f32(out)))
    return out


def _pool(name, fn):
    def op(x, kernel, stride=None, padding="VALID"):
        x = _arr(x, np.float32)
        if x.ndim != 4:
            raise ValueError("%s expects x[n,h,w,c]" % name)
        kh, kw = _pair(kernel)
        sh, sw = _pair(kernel if stride is None else stride)
        pad = _padding(padding)
        n, h, w, c = x.shape
        oh = lib.ptensor_conv_output_size(h, kh, sh, pad)
        ow = lib.ptensor_conv_output_size(w, kw, sw, pad)
        if not oh or not ow:
            raise ValueError("kernel does not fit the input")
        out = np.empty((n, oh, ow, c), np.float32)
        _check(name, fn(_f32(x), n, h, w, c, kh, kw, sh, sw, pad, _f32(out)))
        return out
    op.__name__ = name
    op.__doc__ = "%s over NHWC input (stride defaults to the kernel)." % name
    return op


maxpool2d = _pool("maxpool2d", lib.ptensor_maxpool2d_f32)
avgpool2d = _pool("avgpool2d", lib.ptensor_avgpool2d_f32)


def quantize(x, scale, zero_point):
    """float32 -> int8 affine quantization (TFLite semantics)."""
    x = _arr(x, np.float32)
    out = np.empty(x.shape, np.int8)
    _check("quantize", lib.ptensor_quantize_i8(
        _f32(x), _ptr(out, ctypes.c_int8), x.size, scale, zero_point))
    return out


def dequantize(q, scale, zero_point):
    """int8 -> float32: (q - zero_point) * scale."""
    q = _arr(q, np.int8)
    out = np.empty(q.shape, np.float32)
    _check("dequantize", lib.ptensor_dequantize_i8(
        _ptr(q, ctypes.c_int8), _f32(out), q.size, scale, zero_point))
    return out


def _scales(scale, channels, name):
    """Return (float32 array, count) for a per-tensor or per-channel scale."""
    a = np.ascontiguousarray(np.atleast_1d(scale), dtype=np.float32).reshape(-1)
    if a.size not in (1, channels):
        raise ValueError("%s must be a scalar or have %d entries" % (name, channels))
    if not np.all(a > 0):
        raise ValueError("%s must be positive" % name)
    return a, a.size


def fully_connected_int8(x, x_scale, x_zero_point, weights, w_scale, bias,
                         out_scale, out_zero_point, activation=None):
    """TFLite-compatible int8 fully connected layer.

    weights: int8 [in, out], symmetric (zero point 0).
    w_scale: a scalar (per-tensor) or one scale per output channel, which is
    what TFLite produces for int8 dense layers.
    bias: int32 [out] with scale x_scale * w_scale, or None.
    """
    x = _arr(np.atleast_2d(x), np.int8)
    w = _arr(weights, np.int8)
    if w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("expects x[batch,in] and weights[in,out]")
    b = None if bias is None else _arr(bias, np.int32).reshape(-1)
    if b is not None and b.shape[0] != w.shape[1]:
        raise ValueError("bias must have out_features elements")
    scales, n_scales = _scales(w_scale, w.shape[1], "w_scale")
    out = np.empty((x.shape[0], w.shape[1]), np.int8)
    if n_scales == 1:
        status = lib.ptensor_fully_connected_i8(
            _ptr(x, ctypes.c_int8), x_scale, x_zero_point, _ptr(w, ctypes.c_int8),
            float(scales[0]), _opt_ptr(b, ctypes.c_int32), _ptr(out, ctypes.c_int8),
            out_scale, out_zero_point, x.shape[0], x.shape[1], w.shape[1],
            _activation(activation))
    else:
        status = lib.ptensor_fully_connected_i8_pc(
            _ptr(x, ctypes.c_int8), x_scale, x_zero_point, _ptr(w, ctypes.c_int8),
            _f32(scales), _opt_ptr(b, ctypes.c_int32), _ptr(out, ctypes.c_int8),
            out_scale, out_zero_point, x.shape[0], x.shape[1], w.shape[1],
            _activation(activation))
    _check("fully_connected_int8", status)
    return out


def conv2d_int8(x, x_scale, x_zero_point, filters, f_scale, bias, out_scale,
                out_zero_point, stride=1, padding="SAME", activation=None):
    """TFLite-compatible int8 2D convolution.

    x: int8 NHWC. filters: int8 OHWI, symmetric, with f_scale a scalar or one
    scale per output channel. bias: int32 [out_c] (scale x_scale * f_scale)
    or None.
    """
    x = _arr(x, np.int8)
    f = _arr(filters, np.int8)
    if x.ndim != 4 or f.ndim != 4 or x.shape[3] != f.shape[3]:
        raise ValueError("conv2d_int8 expects x[n,h,w,c] and filters[out_c,kh,kw,c]")
    n, h, w, c = x.shape
    oc, kh, kw, _ = f.shape
    scales, n_scales = _scales(f_scale, oc, "f_scale")
    b = None if bias is None else _arr(bias, np.int32).reshape(-1)
    if b is not None and b.shape[0] != oc:
        raise ValueError("bias must have out_c elements")
    sh, sw = _pair(stride)
    pad = _padding(padding)
    oh = lib.ptensor_conv_output_size(h, kh, sh, pad)
    ow = lib.ptensor_conv_output_size(w, kw, sw, pad)
    if not oh or not ow:
        raise ValueError("filter does not fit the input")
    out = np.empty((n, oh, ow, oc), np.int8)
    _check("conv2d_int8", lib.ptensor_conv2d_i8(
        _ptr(x, ctypes.c_int8), n, h, w, c, x_scale, x_zero_point,
        _ptr(f, ctypes.c_int8), oc, kh, kw, _f32(scales), n_scales,
        _opt_ptr(b, ctypes.c_int32), sh, sw, pad, _activation(activation),
        _ptr(out, ctypes.c_int8), out_scale, out_zero_point))
    return out


def _pool_int8(name, fn, doc):
    def op(x, kernel, stride=None, padding="VALID"):
        x = _arr(x, np.int8)
        if x.ndim != 4:
            raise ValueError("%s expects x[n,h,w,c]" % name)
        kh, kw = _pair(kernel)
        sh, sw = _pair(kernel if stride is None else stride)
        pad = _padding(padding)
        n, h, w, c = x.shape
        oh = lib.ptensor_conv_output_size(h, kh, sh, pad)
        ow = lib.ptensor_conv_output_size(w, kw, sw, pad)
        if not oh or not ow:
            raise ValueError("kernel does not fit the input")
        out = np.empty((n, oh, ow, c), np.int8)
        _check(name, fn(_ptr(x, ctypes.c_int8), n, h, w, c, kh, kw, sh, sw, pad,
                        _ptr(out, ctypes.c_int8)))
        return out
    op.__name__ = name
    op.__doc__ = doc
    return op


maxpool2d_int8 = _pool_int8(
    "maxpool2d_int8", lib.ptensor_maxpool2d_i8,
    "int8 max pooling over NHWC input; output keeps the input's quantization.")
avgpool2d_int8 = _pool_int8(
    "avgpool2d_int8", lib.ptensor_avgpool2d_i8,
    "int8 average pooling over NHWC input (TFLite arithmetic); output keeps the "
    "input's quantization.")


def _act_int8(name, fn, out_scale, out_zp):
    def op(x, x_scale, x_zero_point, out_scale=out_scale, out_zero_point=out_zp):
        x = _arr(x, np.int8)
        out = np.empty_like(x)
        _check(name, fn(_ptr(x, ctypes.c_int8), _ptr(out, ctypes.c_int8), x.size,
                        float(x_scale), int(x_zero_point), float(out_scale),
                        int(out_zero_point)))
        return out
    op.__name__ = name
    op.__doc__ = ("int8 %s, elementwise. The output defaults are TFLite's (scale %g, "
                  "zero point %d)." % (name.split("_")[0], out_scale, out_zp))
    return op


sigmoid_int8 = _act_int8("sigmoid_int8", lib.ptensor_sigmoid_i8, 1.0 / 256, -128)
tanh_int8 = _act_int8("tanh_int8", lib.ptensor_tanh_i8, 1.0 / 128, 0)


def softmax_int8(x, x_scale, x_zero_point, out_scale=1.0 / 256, out_zero_point=-128):
    """int8 softmax over the last axis: dequantize, softmax, requantize.

    The defaults are the output parameters TFLite uses for int8 softmax.
    """
    return quantize(softmax(dequantize(x, x_scale, x_zero_point)),
                    out_scale, out_zero_point)


# --- Extended layer types -------------------------------------------------

_BINARY = {"add": 0, "sub": 1, "mul": 2, "div": 3, "maximum": 4, "minimum": 5,
           "squared_difference": 6, "pow": 7}

_UNARY = {"leaky_relu": 0, "elu": 1, "selu": 2, "hard_sigmoid": 3,
          "hard_swish": 4, "swish": 5, "gelu": 6, "softplus": 7, "softsign": 8,
          "abs": 9, "neg": 10, "square": 11, "sqrt": 12, "rsqrt": 13, "exp": 14,
          "log": 15, "floor": 16, "ceil": 17, "round": 18, "sin": 19, "cos": 20,
          "clip": 21}


def _u32arr(shape):
    a = np.ascontiguousarray(shape if len(shape) else (1,), dtype=np.uint32)
    return a, _ptr(a, ctypes.c_uint32)


def binary(op, a, b):
    """Elementwise binary op with NumPy broadcasting (rank <= 4) in float32.

    op is one of: add, sub, mul, div, maximum, minimum, squared_difference, pow.
    """
    if op not in _BINARY:
        raise ValueError("unknown binary op %r" % (op,))
    a = _arr(a, np.float32)
    b = _arr(b, np.float32)
    shape = np.broadcast_shapes(a.shape, b.shape)
    if max(a.ndim, b.ndim) > 4:
        raise ValueError("binary ops support rank <= 4")
    out = np.empty(shape, np.float32)
    sa, pa = _u32arr(a.shape)
    sb, pb = _u32arr(b.shape)
    _check(op, lib.ptensor_binary_f32(_BINARY[op], _f32(a), pa, max(a.ndim, 1),
                                      _f32(b), pb, max(b.ndim, 1), _f32(out)))
    return out


def _binary(name):
    def op(a, b):
        return binary(name, a, b)
    op.__name__ = name
    op.__doc__ = "Elementwise %s with broadcasting in float32." % name
    return op


sub = _binary("sub")
mul = _binary("mul")
div = _binary("div")
maximum = _binary("maximum")
minimum = _binary("minimum")
squared_difference = _binary("squared_difference")
power = _binary("pow")


def broadcast_add(a, b):
    """a + b with NumPy broadcasting in float32."""
    return binary("add", a, b)


def unary(op, x, p0=0.0, p1=0.0):
    """Elementwise activation / math op in float32 (see UNARY_OPS)."""
    if op not in _UNARY:
        raise ValueError("unknown unary op %r" % (op,))
    x = _arr(x, np.float32)
    out = np.empty_like(x)
    _check(op, lib.ptensor_unary_f32(_UNARY[op], _f32(x), _f32(out), x.size,
                                     float(p0), float(p1)))
    return out


UNARY_OPS = tuple(sorted(_UNARY))


def leaky_relu(x, alpha=0.2):
    return unary("leaky_relu", x, alpha)


def elu(x, alpha=1.0):
    return unary("elu", x, alpha)


def clip(x, lo, hi):
    return unary("clip", x, lo, hi)


def batch_norm(x, mean, variance, gamma, beta, epsilon=1e-3):
    """Inference batch norm over the last axis (channels) in float32."""
    x = _arr(x, np.float32)
    c = x.shape[-1]
    params = [_arr(p, np.float32).reshape(-1) for p in (mean, variance, gamma, beta)]
    if any(p.shape[0] != c for p in params):
        raise ValueError("mean/variance/gamma/beta must have x.shape[-1] elements")
    out = np.empty_like(x)
    _check("batch_norm", lib.ptensor_batchnorm_f32(
        _f32(x), x.size, c, *[_f32(p) for p in params], float(epsilon), _f32(out)))
    return out


def pad2d(x, top, bottom, left, right, value=0.0):
    """Pad the H and W axes of an NHWC tensor with a constant."""
    x = _arr(x, np.float32)
    if x.ndim != 4:
        raise ValueError("pad2d expects x[n,h,w,c]")
    n, h, w, c = x.shape
    out = np.empty((n, h + top + bottom, w + left + right, c), np.float32)
    _check("pad2d", lib.ptensor_pad2d_f32(_f32(x), n, h, w, c, top, bottom,
                                          left, right, float(value), _f32(out)))
    return out


def concat(a, b, axis=-1):
    """Concatenate two float32 tensors of equal rank (<= 4) along axis."""
    a = _arr(a, np.float32)
    b = _arr(b, np.float32)
    if a.ndim != b.ndim or not 1 <= a.ndim <= 4:
        raise ValueError("concat expects two tensors of the same rank (1..4)")
    ax = axis + a.ndim if axis < 0 else axis
    shape = list(a.shape)
    shape[ax] += b.shape[ax]
    out = np.empty(shape, np.float32)
    sa, pa = _u32arr(a.shape)
    sb, pb = _u32arr(b.shape)
    _check("concat", lib.ptensor_concat_f32(_f32(a), pa, _f32(b), pb, a.ndim,
                                            axis, _f32(out)))
    return out


def global_avgpool2d(x):
    """Mean over H and W of an NHWC tensor, giving [n, c]."""
    x = _arr(x, np.float32)
    if x.ndim != 4:
        raise ValueError("global_avgpool2d expects x[n,h,w,c]")
    n, h, w, c = x.shape
    out = np.empty((n, c), np.float32)
    _check("global_avgpool2d", lib.ptensor_global_avgpool_f32(_f32(x), n, h, w, c, _f32(out)))
    return out


def depthwise_conv2d(x, filters, bias=None, stride=1, padding="SAME",
                     depth_multiplier=1, dilation=1):
    """Depthwise conv, x: NHWC, filters: [1, kh, kw, c * depth_multiplier]."""
    x = _arr(x, np.float32)
    f = _arr(filters, np.float32)
    if x.ndim != 4 or f.ndim != 4 or f.shape[0] != 1:
        raise ValueError("depthwise_conv2d expects x[n,h,w,c] and filters[1,kh,kw,c*m]")
    n, h, w, c = x.shape
    _, kh, kw, oc = f.shape
    m = int(depth_multiplier)
    if oc != c * m:
        raise ValueError("filters.shape[3] must be c * depth_multiplier")
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    pad = _padding(padding)
    oh = lib.ptensor_conv_output_size(h, (kh - 1) * dh + 1, sh, pad)
    ow = lib.ptensor_conv_output_size(w, (kw - 1) * dw + 1, sw, pad)
    if not oh or not ow:
        raise ValueError("kernel does not fit the input")
    b = None if bias is None else _arr(bias, np.float32).reshape(-1)
    if b is not None and b.shape[0] != oc:
        raise ValueError("bias must have c * depth_multiplier elements")
    out = np.empty((n, oh, ow, oc), np.float32)
    _check("depthwise_conv2d", lib.ptensor_depthwise_conv2d_f32(
        _f32(x), n, h, w, c, _f32(f), kh, kw, m, _opt_ptr(b, ctypes.c_float),
        sh, sw, pad, dh, dw, _f32(out)))
    return out


def lstm_cell(x, h, c, W, U, b):
    """One LSTM step (Keras layout, gates i, f, c, o). Returns (h_new, c_new).

    x[batch,in], h/c[batch,hidden], W[in,4*hidden], U[hidden,4*hidden], b[4*hidden].
    """
    x, h, c, W, U = (_arr(np.atleast_2d(t), np.float32) for t in (x, h, c, W, U))
    b = _arr(b, np.float32).reshape(-1)
    batch, nin = x.shape
    hidden = h.shape[1]
    if W.shape != (nin, 4 * hidden) or U.shape != (hidden, 4 * hidden) or \
            b.shape[0] != 4 * hidden or c.shape != h.shape or h.shape[0] != batch:
        raise ValueError("lstm_cell shapes do not match the Keras layout")
    h_out = np.empty_like(h)
    c_out = np.empty_like(c)
    _check("lstm_cell", lib.ptensor_lstm_cell_f32(
        _f32(x), _f32(h), _f32(c), _f32(W), _f32(U), _f32(b), batch, nin, hidden,
        _f32(h_out), _f32(c_out)))
    return h_out, c_out


def gru_cell(x, h, W, U, bx, bh):
    """One GRU step (Keras reset_after=True layout, gates z, r, h).

    x[batch,in], h[batch,hidden], W[in,3*hidden], U[hidden,3*hidden],
    bx/bh[3*hidden] (the two rows of Keras' bias).
    """
    x, h, W, U = (_arr(np.atleast_2d(t), np.float32) for t in (x, h, W, U))
    bx = _arr(bx, np.float32).reshape(-1)
    bh = _arr(bh, np.float32).reshape(-1)
    batch, nin = x.shape
    hidden = h.shape[1]
    if W.shape != (nin, 3 * hidden) or U.shape != (hidden, 3 * hidden) or \
            bx.shape[0] != 3 * hidden or bh.shape[0] != 3 * hidden:
        raise ValueError("gru_cell shapes do not match the Keras layout")
    h_out = np.empty_like(h)
    _check("gru_cell", lib.ptensor_gru_cell_f32(
        _f32(x), _f32(h), _f32(W), _f32(U), _f32(bx), _f32(bh), batch, nin,
        hidden, _f32(h_out)))
    return h_out
