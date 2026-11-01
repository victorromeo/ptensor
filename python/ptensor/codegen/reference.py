"""NumPy interpreter for float graphs.

Used to check passes (a fused or pruned graph must compute what the original
did), to collect activation ranges for int8 calibration, and as the expected
output when generated C is tested. Weights with an ``encoding`` are decoded
exactly as the device decodes them, so the result is what the generated code
should produce.
"""
import numpy as np

from .ir import FLOAT32, clamp_range


def weight(graph, name):
    t = graph.tensors[name]
    if t.encoding is not None:
        return np.asarray(t.encoding.decode(), np.float32).reshape(t.shape)
    return t.data.astype(np.float32)


def _pad_amounts(size, out, k, s, d, padding):
    if padding == "VALID":
        return 0
    eff = (k - 1) * d + 1
    total = max(0, (out - 1) * s + eff - size)
    return total // 2


def _patches(x, kh, kw, sh, sw, dh, dw, oh, ow, padding, fill=0.0):
    """[oh, ow, kh, kw, C] windows of an [H, W, C] array."""
    h, w, c = x.shape
    pt = _pad_amounts(h, oh, kh, sh, dh, padding)
    pl = _pad_amounts(w, ow, kw, sw, dw, padding)
    ph = max(0, (oh - 1) * sh + (kh - 1) * dh + 1 - h - pt)
    pw = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - w - pl)
    xp = np.full((h + pt + ph, w + pl + pw, c), fill, x.dtype)
    xp[pt:pt + h, pl:pl + w] = x
    out = np.empty((oh, ow, kh, kw, c), x.dtype)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i, j] = xp[i * dh: i * dh + (oh - 1) * sh + 1: sh,
                                 j * dw: j * dw + (ow - 1) * sw + 1: sw]
    valid = np.zeros((oh, ow, kh, kw), bool)
    mask = np.zeros((h + pt + ph, w + pl + pw), bool)
    mask[pt:pt + h, pl:pl + w] = True
    for i in range(kh):
        for j in range(kw):
            valid[:, :, i, j] = mask[i * dh: i * dh + (oh - 1) * sh + 1: sh,
                                     j * dw: j * dw + (ow - 1) * sw + 1: sw]
    return out, valid


def _act(y, activation):
    if activation is None:
        return y
    lo, hi = clamp_range(activation)
    return np.clip(y, lo, hi)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _erf(x):
    try:
        from math import erf
    except ImportError:  # pragma: no cover
        raise
    return np.vectorize(erf, otypes=[np.float64])(x)


def unary(kind, x, attrs=None):
    """Elementwise activation / math function `kind` (ir.UNARY_KINDS)."""
    a = attrs or {}
    x = np.asarray(x, np.float64)
    if kind in ("relu", "relu6", "relu1"):
        return _act(x, kind)
    f = {
        "clip": lambda v: np.clip(v, a.get("lo"), a.get("hi")),
        "sigmoid": _sigmoid,
        "tanh": np.tanh,
        "leaky_relu": lambda v: np.where(v >= 0, v, a.get("alpha", 0.01) * v),
        "elu": lambda v: np.where(v >= 0, v, a.get("alpha", 1.0) * np.expm1(np.minimum(v, 0))),
        "celu": lambda v: np.where(v >= 0, v, a.get("alpha", 1.0) *
                                   np.expm1(np.minimum(v, 0) / a.get("alpha", 1.0))),
        "selu": lambda v: 1.0507009873554805 * np.where(
            v >= 0, v, 1.6732632423543772 * np.expm1(np.minimum(v, 0))),
        "gelu": lambda v: 0.5 * v * (1.0 + _erf(v / np.sqrt(2.0))),
        "gelu_tanh": lambda v: 0.5 * v * (1.0 + np.tanh(0.7978845608028654 *
                                                        (v + 0.044715 * v ** 3))),
        "silu": lambda v: v * _sigmoid(v),
        "mish": lambda v: v * np.tanh(_softplus(v)),
        "hard_sigmoid": lambda v: np.clip(v / 6.0 + 0.5, 0.0, 1.0),
        "hard_swish": lambda v: v * np.clip(v / 6.0 + 0.5, 0.0, 1.0),
        "softplus": _softplus,
        "softsign": lambda v: v / (1.0 + np.abs(v)),
        "log_sigmoid": lambda v: -_softplus(-v),
        "abs": np.abs, "neg": np.negative, "square": np.square,
        "sqrt": np.sqrt, "rsqrt": lambda v: 1.0 / np.sqrt(v), "exp": np.exp, "log": np.log,
        "floor": np.floor, "ceil": np.ceil, "round": np.round, "sin": np.sin, "cos": np.cos,
    }.get(kind)
    if f is None:
        raise NotImplementedError("activation %s" % kind)
    with np.errstate(all="ignore"):
        return f(x)


BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "maximum": np.maximum, "minimum": np.minimum,
    "squared_difference": lambda p, q: (p - q) ** 2, "pow": np.power,
}


def _value(graph, env, name):
    return env[name] if name in env else weight(graph, name)


def _lstm(x, W, U, b, H):
    h = np.zeros(H)
    c = np.zeros(H)
    out = []
    for t in range(x.shape[1]):
        z = x[0, t] @ W + h @ U + b
        i, f, g, o = _sigmoid(z[:H]), _sigmoid(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), \
            _sigmoid(z[3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        out.append(h)
    return np.stack(out)[None]


def _gru(x, W, U, bx, bh, H):
    h = np.zeros(H)
    out = []
    for t in range(x.shape[1]):
        xz = x[0, t] @ W + bx
        hz = h @ U + bh
        z = _sigmoid(xz[:H] + hz[:H])
        r = _sigmoid(xz[H:2 * H] + hz[H:2 * H])
        n = np.tanh(xz[2 * H:] + r * hz[2 * H:])
        h = z * h + (1.0 - z) * n
        out.append(h)
    return np.stack(out)[None]


def run_node(graph, node, env):
    a = node.attrs
    x = _value(graph, env, node.inputs[0]) if node.inputs else None
    out = graph.tensors[node.outputs[0]]
    op = node.op
    if op in ("conv2d", "depthwise_conv2d"):
        f = weight(graph, node.inputs[1])
        b = weight(graph, node.inputs[2])
        (sh, sw), (dh, dw) = a["stride"], a.get("dilation", (1, 1))
        oh, ow = out.shape[1], out.shape[2]
        pat, _ = _patches(x[0], f.shape[1], f.shape[2], sh, sw, dh, dw, oh, ow, a["padding"])
        if op == "conv2d":
            y = np.einsum("hwijc,oijc->hwo", pat.astype(np.float64), f.astype(np.float64))
        else:
            m = a.get("depth_multiplier", 1)
            pat = np.repeat(pat, m, axis=-1)
            y = np.einsum("hwijc,ijc->hwc", pat.astype(np.float64), f[0].astype(np.float64))
        y = (y + b).astype(np.float32)[None]
        return _act(y, a.get("activation"))
    if op == "dense":
        w = weight(graph, node.inputs[1])
        b = weight(graph, node.inputs[2])
        y = (x.reshape(1, -1).astype(np.float64) @ w.astype(np.float64) + b).astype(np.float32)
        return _act(y, a.get("activation"))
    if op in ("max_pool2d", "avg_pool2d"):
        (kh, kw), (sh, sw) = a["kernel"], a["stride"]
        oh, ow = out.shape[1], out.shape[2]
        fill = -np.inf if op == "max_pool2d" else 0.0
        pat, valid = _patches(x[0], kh, kw, sh, sw, 1, 1, oh, ow, a["padding"], fill)
        if op == "max_pool2d":
            y = pat.max(axis=(2, 3))
        else:
            cnt = valid.sum(axis=(2, 3))[..., None]
            y = pat.sum(axis=(2, 3)) / np.maximum(cnt, 1)
        return y.astype(np.float32)[None]
    if op == "global_avg_pool":
        return x.mean(axis=(1, 2)).reshape(out.shape).astype(np.float32)
    if op == "batch_norm":
        mean, var, gamma, beta = (weight(graph, n) for n in node.inputs[1:5])
        inv = gamma / np.sqrt(var + a.get("epsilon", 1e-3))
        return ((x - mean) * inv + beta).astype(np.float32)
    if op == "activation":
        return unary(a["kind"], x, a).astype(np.float32)
    if op == "add":
        y = x.astype(np.float64) + _value(graph, env, node.inputs[1])
        return _act(y.astype(np.float32).reshape(out.shape), a.get("activation"))
    if op == "binary":
        p = _value(graph, env, node.inputs[0]).astype(np.float64)
        q = _value(graph, env, node.inputs[1]).astype(np.float64)
        with np.errstate(all="ignore"):
            return BINARY[a["kind"]](p, q).astype(np.float32).reshape(out.shape)
    if op == "reshape":
        return x.reshape(out.shape)
    if op == "softmax":
        z = x.astype(np.float64) * a.get("beta", 1.0)
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32).reshape(out.shape)
    if op == "pad":
        t, bo, l, r = a["pads"]
        return np.pad(x, ((0, 0), (t, bo), (l, r), (0, 0)), constant_values=a.get("value", 0.0))
    if op == "concat":
        return np.concatenate([_value(graph, env, i) for i in node.inputs], axis=a["axis"])
    if op == "transpose":
        return np.ascontiguousarray(np.transpose(x, a["perm"]))
    if op == "mean":
        return x.astype(np.float64).mean(axis=tuple(a["axes"])).astype(np.float32).reshape(out.shape)
    if op == "slice":
        idx = tuple(slice(b0, b0 + (n - 1) * st + 1, st)
                    for b0, st, n in zip(a["begin"], a["step"], out.shape))
        return np.ascontiguousarray(x[idx])
    if op == "matmul":
        q = _value(graph, env, node.inputs[1])
        return (x.astype(np.float64) @ q.astype(np.float64)).astype(np.float32)
    if op == "lstm":
        W, U, b = (weight(graph, n).astype(np.float64) for n in node.inputs[1:4])
        return _lstm(x.astype(np.float64), W, U, b, U.shape[0]).astype(np.float32)
    if op == "gru":
        W, U, bx, bh = (weight(graph, n).astype(np.float64) for n in node.inputs[1:5])
        return _gru(x.astype(np.float64), W, U, bx, bh, U.shape[0]).astype(np.float32)
    raise NotImplementedError("reference interpreter has no %s" % op)


def run(graph, inputs, collect=False):
    """Run a float graph on one example (batch 1).

    inputs: an array, or a list / dict of arrays matching graph.inputs.
    Returns the list of output arrays, plus (with collect=True) a dict of
    every activation's value.
    """
    if graph.is_quantized():
        raise ValueError("the reference interpreter runs float graphs only")
    if isinstance(inputs, dict):
        feeds = [inputs[n] for n in graph.inputs]
    elif isinstance(inputs, (list, tuple)):
        feeds = list(inputs)
    else:
        feeds = [inputs]
    env = {}
    for name, v in zip(graph.inputs, feeds):
        t = graph.tensors[name]
        env[name] = np.asarray(v, np.float32).reshape(t.shape)
    for node in graph.nodes:
        env[node.outputs[0]] = run_node(graph, node, env)
    outs = [env[n] for n in graph.outputs]
    return (outs, env) if collect else outs


def run_batch(graph, x):
    """Run a single-input, single-output float graph on each row of x."""
    return np.stack([run(graph, xi)[0].reshape(-1) for xi in x])


def run_ptensor_int8(graph, x):
    """Run an int8 graph through pTensor's own kernels via the Python
    bindings (``ptensor.conv2d_int8`` and friends): what the generated code
    must reproduce bit for bit. Supports conv2d, dense, max_pool2d, avg_pool2d,
    reshape, softmax, add, relu / relu6, sigmoid, tanh, quantize and dequantize."""
    import ptensor
    from ptensor import fusion

    def qp(name):
        q = graph.tensors[name].quant
        return float(q.scale[0]), int(q.zero_point[0])

    env = {graph.inputs[0]: np.asarray(x).reshape(graph.tensors[graph.inputs[0]].shape)}
    for n in graph.nodes:
        a, x0, y = n.attrs, env[n.inputs[0]], n.outputs[0]
        out_shape = graph.tensors[y].shape
        if n.op == "conv2d":
            w, b = graph.tensors[n.inputs[1]], graph.tensors[n.inputs[2]]
            if tuple(a.get("dilation", (1, 1))) != (1, 1):
                raise NotImplementedError("dilated int8 conv in the bindings")
            v = ptensor.conv2d_int8(x0, *qp(n.inputs[0]), filters=w.data, f_scale=w.quant.scale,
                                    bias=b.data, out_scale=qp(y)[0], out_zero_point=qp(y)[1],
                                    stride=tuple(a["stride"]), padding=a["padding"],
                                    activation=a.get("activation"))
        elif n.op == "dense":
            w, b = graph.tensors[n.inputs[1]], graph.tensors[n.inputs[2]]
            v = ptensor.fully_connected_int8(x0.reshape(1, -1), *qp(n.inputs[0]), weights=w.data,
                                             w_scale=w.quant.scale, bias=b.data,
                                             out_scale=qp(y)[0], out_zero_point=qp(y)[1],
                                             activation=a.get("activation"))
        elif n.op == "max_pool2d":
            v = ptensor.maxpool2d_int8(x0, tuple(a["kernel"]), tuple(a["stride"]), a["padding"])
        elif n.op == "avg_pool2d":
            v = ptensor.avgpool2d_int8(x0, tuple(a["kernel"]), tuple(a["stride"]), a["padding"])
        elif n.op == "activation" and a["kind"] in ("sigmoid", "tanh"):
            fn = ptensor.sigmoid_int8 if a["kind"] == "sigmoid" else ptensor.tanh_int8
            v = fn(x0, *(qp(n.inputs[0]) + qp(y)))
        elif n.op in ("reshape", "transpose", "slice", "pad"):
            v = run_node(graph, n, {n.inputs[0]: x0})
            if n.op == "pad":
                v = v.astype(np.int8)
        elif n.op == "softmax":
            v = ptensor.softmax_int8(x0.reshape(1, -1), *qp(n.inputs[0]), *qp(y))
        elif n.op == "add":
            v = fusion.add_int8(x0, *qp(n.inputs[0]), env[n.inputs[1]], *qp(n.inputs[1]), *qp(y),
                                activation=a.get("activation"))
        elif n.op == "activation" and a["kind"] in ("relu", "relu6"):
            lo = qp(y)[1]
            hi = 127 if a["kind"] == "relu" else min(127, int(round(6.0 / qp(y)[0])) + lo)
            v = np.clip(x0, lo, hi).astype(np.int8)
        elif n.op == "quantize":
            v = ptensor.quantize(x0, *qp(y))
        elif n.op == "dequantize":
            v = ptensor.dequantize(x0, *qp(n.inputs[0]))
        else:
            raise NotImplementedError("int8 %s in the bindings runner" % n.op)
        env[y] = np.asarray(v).reshape(out_shape)
    return [env[o] for o in graph.outputs]


__all__ = ["run", "run_batch", "run_node", "run_ptensor_int8", "unary", "weight", "FLOAT32"]
