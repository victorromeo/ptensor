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


def fully_connected_int8(x, x_scale, x_zero_point, weights, w_scale, bias,
                         out_scale, out_zero_point, activation=None):
    """TFLite-compatible int8 fully connected layer.

    weights: int8 [in, out], symmetric per-tensor (zero point 0).
    bias: int32 [out] with scale x_scale * w_scale, or None.
    """
    x = _arr(np.atleast_2d(x), np.int8)
    w = _arr(weights, np.int8)
    if w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("expects x[batch,in] and weights[in,out]")
    b = None if bias is None else _arr(bias, np.int32).reshape(-1)
    out = np.empty((x.shape[0], w.shape[1]), np.int8)
    _check("fully_connected_int8", lib.ptensor_fully_connected_i8(
        _ptr(x, ctypes.c_int8), x_scale, x_zero_point, _ptr(w, ctypes.c_int8),
        w_scale, _opt_ptr(b, ctypes.c_int32), _ptr(out, ctypes.c_int8),
        out_scale, out_zero_point, x.shape[0], x.shape[1], w.shape[1],
        _activation(activation)))
    return out
