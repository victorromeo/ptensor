"""Framework-neutral graph that every frontend produces and every pass and
emitter consumes.

Conventions (the layouts pTensor's kernels use, so emitters never transpose):

* Activations are NHWC with batch 1. Rank 2 tensors are [1, features].
* conv2d filters are OHWI ``[out_c, kh, kw, in_c]``.
* depthwise_conv2d filters are ``[1, kh, kw, in_c * multiplier]``.
* dense weights are ``[in_features, out_features]``.
* Every conv2d / depthwise_conv2d / dense has a bias (frontends add a zero
  bias when the source model has none), so emitters need no special case.

Operators (``Node.op``) and their attributes:

=================  ===================================  ============================================
op                 inputs                               attrs
=================  ===================================  ============================================
conv2d             x, filter, bias                      stride, padding, dilation, activation
depthwise_conv2d   x, filter, bias                      stride, padding, dilation, depth_multiplier,
                                                        activation
dense              x, weights, bias                     activation
max_pool2d         x                                    kernel, stride, padding
avg_pool2d         x                                    kernel, stride, padding
global_avg_pool    x                                    (output [1, C] or [1, 1, 1, C])
batch_norm         x, mean, variance, gamma, beta       epsilon
activation         x                                    kind: relu, relu6, relu1, clip, sigmoid, tanh;
                                                        lo, hi for clip
add                a, b                                 activation
reshape            x                                    (output shape is the target)
softmax            x                                    beta
quantize           x (float)                            (output tensor holds the int8 parameters)
dequantize         x (int8)                             --
=================  ===================================  ============================================

``activation`` on conv2d / depthwise_conv2d / dense / add is a fused clamp:
None, "relu", "relu6", "relu1", or a ``(lo, hi)`` pair. Padding is "SAME" or
"VALID". Strides, kernels and dilations are (h, w) pairs.

Tensors carry an optional ``quant`` (scale / zero point arrays, per tensor or
per channel along ``axis``) for int8 / int32 data, and constants may carry an
``encoding`` (a ``ptensor.qat.Encoded``: packed, Hough or polynomial codes)
that says how an emitter must store them.
"""
import copy

import numpy as np

FLOAT32, INT8, INT32 = "float32", "int8", "int32"
DTYPES = {FLOAT32: np.float32, INT8: np.int8, INT32: np.int32}

LINEAR_OPS = ("conv2d", "depthwise_conv2d", "dense")
ACTIVATION_RANGES = {
    None: (-np.inf, np.inf), "relu": (0.0, np.inf), "relu6": (0.0, 6.0),
    "relu1": (-1.0, 1.0),
}
OPS = LINEAR_OPS + ("max_pool2d", "avg_pool2d", "global_avg_pool", "batch_norm",
                    "activation", "add", "reshape", "softmax", "quantize",
                    "dequantize")


def clamp_range(activation):
    """(lo, hi) of a fused activation: None, a name, or a (lo, hi) pair."""
    if isinstance(activation, (tuple, list)):
        return float(activation[0]), float(activation[1])
    if activation not in ACTIVATION_RANGES:
        raise ValueError("not a clamp activation: %r" % (activation,))
    return ACTIVATION_RANGES[activation]


def canonical_activation(activation):
    """Name for a clamp range when it has one, else the (lo, hi) pair."""
    lo, hi = clamp_range(activation)
    for name, rng in ACTIVATION_RANGES.items():
        if rng == (lo, hi):
            return name
    return (lo, hi)


class QuantParams(object):
    """Affine quantisation: real = (q - zero_point) * scale.

    Per tensor when ``scale`` has one element, else per channel along ``axis``.
    """

    def __init__(self, scale, zero_point, axis=None):
        self.scale = np.atleast_1d(np.asarray(scale, np.float32))
        self.zero_point = np.atleast_1d(np.asarray(zero_point, np.int32))
        if self.zero_point.size == 1 and self.scale.size > 1:
            self.zero_point = np.repeat(self.zero_point, self.scale.size)
        self.axis = axis

    @property
    def per_channel(self):
        return self.scale.size > 1

    def __repr__(self):
        if self.per_channel:
            return "QuantParams(%d channels, axis=%s)" % (self.scale.size, self.axis)
        return "QuantParams(scale=%g, zp=%d)" % (self.scale[0], self.zero_point[0])


class Tensor(object):
    """A value in the graph. Constants have ``data``; activations do not."""

    def __init__(self, name, shape, dtype=FLOAT32, data=None, quant=None):
        if dtype not in DTYPES:
            raise ValueError("unsupported dtype %r for tensor %s" % (dtype, name))
        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.data = None if data is None else np.ascontiguousarray(data, DTYPES[dtype])
        if self.data is not None and self.data.shape != self.shape:
            self.data = self.data.reshape(self.shape)
        self.quant = quant
        self.encoding = None  # ptensor.qat.Encoded, set by passes.encode_weights

    @property
    def is_const(self):
        return self.data is not None

    @property
    def size(self):
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self):
        return self.size * np.dtype(DTYPES[self.dtype]).itemsize

    def __repr__(self):
        kind = "const" if self.is_const else "act"
        return "Tensor(%s, %s, %s, %s)" % (self.name, self.shape, self.dtype, kind)


class Node(object):
    """An operator: reads ``inputs`` and writes ``outputs`` (tensor names)."""

    def __init__(self, op, inputs, outputs, attrs=None, name=None):
        if op not in OPS:
            raise ValueError("unknown op %r" % (op,))
        self.op = op
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attrs = dict(attrs or {})
        self.name = name or op

    def __repr__(self):
        return "Node(%s %s: %s -> %s)" % (self.op, self.name, self.inputs, self.outputs)


class Graph(object):
    """Tensors plus nodes in execution order."""

    def __init__(self, name="model"):
        self.name = name
        self.tensors = {}
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.source = None  # e.g. "keras", "tflite"
        self.notes = []     # what the frontend / passes did, for the report

    # -- building -----------------------------------------------------------

    def unique_name(self, base):
        base = base or "t"
        name, i = base, 1
        while name in self.tensors:
            name = "%s_%d" % (base, i)
            i += 1
        return name

    def add_tensor(self, name, shape, dtype=FLOAT32, data=None, quant=None):
        name = self.unique_name(name)
        t = Tensor(name, shape, dtype, data, quant)
        self.tensors[name] = t
        return t

    def add_const(self, name, data, dtype=None, quant=None):
        data = np.asarray(data)
        if dtype is None:
            dtype = {np.dtype(np.int8): INT8, np.dtype(np.int32): INT32}.get(data.dtype, FLOAT32)
        return self.add_tensor(name, data.shape, dtype, data, quant)

    def add_node(self, op, inputs, outputs, attrs=None, name=None):
        node = Node(op, [getattr(t, "name", t) for t in inputs],
                    [getattr(t, "name", t) for t in outputs], attrs, name)
        self.nodes.append(node)
        return node

    # -- queries ------------------------------------------------------------

    def tensor(self, name):
        return self.tensors[name]

    def producer(self, name):
        for n in self.nodes:
            if name in n.outputs:
                return n
        return None

    def consumers(self, name):
        return [n for n in self.nodes if name in n.inputs]

    def activations(self):
        """Non-constant tensors that some node reads or writes, in first-use order."""
        seen, out = set(), []
        for name in self.inputs:
            if name not in seen:
                seen.add(name)
                out.append(name)
        for n in self.nodes:
            for name in n.inputs + n.outputs:
                if name not in seen and not self.tensors[name].is_const:
                    seen.add(name)
                    out.append(name)
        return out

    def constants(self):
        seen, out = set(), []
        for n in self.nodes:
            for name in n.inputs:
                if name not in seen and self.tensors[name].is_const:
                    seen.add(name)
                    out.append(name)
        return out

    def ops_used(self):
        return sorted(set(n.op for n in self.nodes))

    def is_quantized(self):
        return any(self.tensors[t].dtype == INT8 for t in self.activations())

    def copy(self):
        return copy.deepcopy(self)

    # -- editing ------------------------------------------------------------

    def rename_uses(self, old, new):
        """Make every reader of tensor `old` read `new` instead."""
        for n in self.nodes:
            n.inputs = [new if i == old else i for i in n.inputs]
        self.outputs = [new if o == old else o for o in self.outputs]

    def remove_unused(self):
        """Drop tensors nothing reads or writes, and nodes whose outputs are dead."""
        changed = True
        while changed:
            changed = False
            live = set(self.outputs)
            for n in self.nodes:
                live.update(n.inputs)
            keep = [n for n in self.nodes if any(o in live for o in n.outputs)]
            if len(keep) != len(self.nodes):
                self.nodes = keep
                changed = True
        used = set(self.inputs) | set(self.outputs)
        for n in self.nodes:
            used.update(n.inputs)
            used.update(n.outputs)
        self.tensors = {k: v for k, v in self.tensors.items() if k in used}

    def validate(self):
        """Check every tensor exists, is written before it is read, and the
        graph's ops are canonical. Raises ValueError."""
        written = set(self.inputs)
        for n in self.nodes:
            for i in n.inputs:
                if i not in self.tensors:
                    raise ValueError("%s reads unknown tensor %s" % (n, i))
                if not self.tensors[i].is_const and i not in written:
                    raise ValueError("%s reads %s before it is written" % (n, i))
            for o in n.outputs:
                if o not in self.tensors:
                    raise ValueError("%s writes unknown tensor %s" % (n, o))
                written.add(o)
        for o in self.outputs:
            if o not in written:
                raise ValueError("graph output %s is never written" % o)
        return self

    # -- report -------------------------------------------------------------

    def summary(self):
        lines = ["graph %s (from %s): %d nodes" % (self.name, self.source, len(self.nodes))]
        for name in self.inputs:
            t = self.tensors[name]
            lines.append("  input  %-24s %s %s" % (name, t.shape, t.dtype))
        for n in self.nodes:
            out = self.tensors[n.outputs[0]]
            extra = []
            act = n.attrs.get("activation")
            if act is not None:
                extra.append("act=%s" % (act,))
            if n.op == "activation":
                extra.append(n.attrs["kind"])
            for i in n.inputs:
                enc = self.tensors[i].encoding
                if enc is not None:
                    extra.append("w=%s" % enc.describe())
            lines.append("  %-17s %-22s -> %s %s %s" % (n.op, n.name, out.shape, out.dtype,
                                                        " ".join(extra)))
        for name in self.outputs:
            t = self.tensors[name]
            lines.append("  output %-24s %s %s" % (name, t.shape, t.dtype))
        return "\n".join(lines)


def conv_out_size(size, kernel, stride, padding, dilation=1):
    eff = (kernel - 1) * dilation + 1
    if padding == "SAME":
        return (size + stride - 1) // stride
    return (size - eff) // stride + 1
