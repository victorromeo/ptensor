"""PyTorch models, imported through ``torch.export``.

    gen = CodeGenerator(model, example_input=torch.randn(1, 3, 32, 32))

``model`` is an ``nn.Module`` (exported here in eval mode with the example
input), an ``ExportedProgram``, or a ``.pt2`` file saved by
``torch.export.save``. The importer reads the ATen graph, so modules,
``torch.nn.functional`` calls and tensor methods all arrive as the same set of
operators, and anything computed only from constants (``torch.zeros``,
weight reshapes, ...) is folded at import time.

Layout. PyTorch is NCHW; pTensor is NHWC. Every value keeps a permutation
saying how its pTensor tensor relates to the PyTorch one, so convolutions and
pools run on NHWC data, permutes are free, and a transpose is only emitted
where a PyTorch operator really depends on the element order (a view of a
4D tensor, for instance). A flatten feeding a Linear permutes the Linear's
rows instead of the data. With ``channels_last_io=True`` (the default) a 4D
model input or output is NHWC in the generated function; pass
``channels_last_io=False`` to keep PyTorch's NCHW order there (a transpose
is added at the boundary).

Coverage: see ``SUPPORTED`` below and docs/code_generation.md. Operators
pTensor has no kernel for (transposed convolution, upsampling, embedding,
comparisons, ...) raise NotImplementedError naming the operator and node.
"""
import operator

import numpy as np

from ..ir import Graph, conv_out_size
from .base import Frontend, FrontendUnavailable, register_frontend

NHWC = (0, 2, 3, 1)  # pTensor tensor = torch tensor.permute(NHWC)

_UNARY = {
    "relu": "relu", "relu6": "relu6", "sigmoid": "sigmoid", "tanh": "tanh", "silu": "silu",
    "mish": "mish", "hardsigmoid": "hard_sigmoid", "hardswish": "hard_swish",
    "selu": "selu", "log_sigmoid": "log_sigmoid", "log_sigmoid_forward": "log_sigmoid",
    "abs": "abs", "neg": "neg", "negative": "neg", "square": "square", "sqrt": "sqrt",
    "rsqrt": "rsqrt", "exp": "exp", "log": "log", "floor": "floor", "ceil": "ceil",
    "round": "round", "sin": "sin", "cos": "cos",
}
_BINARY = {"add": "add", "sub": "sub", "subtract": "sub", "mul": "mul", "multiply": "mul",
           "div": "div", "divide": "div", "true_divide": "div", "maximum": "maximum",
           "minimum": "minimum", "pow": "pow"}
_IDENTITY = ("contiguous", "clone", "alias", "detach", "dropout", "feature_dropout",
             "alpha_dropout", "feature_alpha_dropout", "lift_fresh_copy", "_to_copy", "to",
             "type_as", "native_dropout", "_assert_tensor_metadata")

#: ATen operator names the importer maps (1d / 2d forms, in-place and
#: functional variants included).
SUPPORTED = sorted(set(list(_UNARY) + list(_BINARY) + list(_IDENTITY) + [
    "conv1d", "conv2d", "convolution", "linear", "addmm", "mm", "matmul", "bmm",
    "batch_norm", "_native_batch_norm_legit_no_training", "layer_norm", "native_layer_norm",
    "group_norm", "native_group_norm", "instance_norm", "max_pool1d", "max_pool2d",
    "max_pool2d_with_indices", "avg_pool1d", "avg_pool2d", "adaptive_avg_pool1d",
    "adaptive_avg_pool2d", "_adaptive_avg_pool2d", "adaptive_max_pool1d",
    "adaptive_max_pool2d", "flatten", "view", "reshape", "_unsafe_view", "squeeze",
    "unsqueeze", "permute", "transpose", "t", "numpy_T", "cat", "concat", "stack", "select",
    "slice", "narrow", "split", "split_with_sizes", "chunk", "unbind", "mean", "sum",
    "softmax", "_softmax", "log_softmax", "_log_softmax", "pad", "constant_pad_nd",
    "hardtanh", "clamp", "clip", "clamp_min", "clamp_max", "leaky_relu", "elu", "celu",
    "gelu", "softplus", "prelu", "_prelu_kernel", "rsub", "reciprocal", "lstm", "gru",
    "sym_size", "getitem", "scaled_dot_product_attention", "amax", "amin", "max", "min",
    "var", "std", "repeat", "expand", "upsample_nearest1d", "upsample_nearest2d",
    "_upsample_nearest_exact1d", "_upsample_nearest_exact2d", "unflatten"]))


class _Value(object):
    """An activation: IR tensor ``name`` = torch value.permute(``perm``).

    ``perm`` lists PyTorch dims in IR order and may leave out dims of size 1
    (so an unsqueeze costs nothing and rank-5 views of rank-4 data work).

    ``flat`` = (C, H, W) marks the result of flattening an NHWC tensor that
    has not been reordered yet: the IR tensor holds the features in H, W, C
    order, while PyTorch's are in C, H, W order.
    """
    __slots__ = ("name", "perm", "shape", "flat")

    def __init__(self, name, perm, shape, flat=None):
        self.name, self.perm, self.shape, self.flat = name, tuple(perm), tuple(shape), flat

    @property
    def rank(self):
        return len(self.shape)

    @property
    def ir_shape(self):
        if self.flat is not None:
            return self.shape
        return tuple(self.shape[p] for p in self.perm) or (1,)


class _Unavailable(object):
    def __init__(self, why):
        self.why = why


def _ident(n):
    return tuple(range(n))


def _pair(v, n=2):
    if isinstance(v, (list, tuple)):
        v = list(v)
        return tuple(int(x) for x in (v * n if len(v) == 1 else v))
    return (int(v),) * n


@register_frontend
class PyTorchFrontend(Frontend):
    name = "pytorch"
    extensions = (".pt2",)

    def accepts(self, source):
        if isinstance(source, str):
            return Frontend.accepts(self, source)
        mods = [c.__module__ for c in type(source).__mro__]
        return any(m.startswith("torch.nn") or m.startswith("torch.export") for m in mods)

    def load(self, source, example_input=None, channels_last_io=True, **options):
        try:
            import torch
        except ImportError:
            raise FrontendUnavailable("importing a PyTorch model needs torch")
        if isinstance(source, str):
            ep = torch.export.load(source)
            name = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        elif isinstance(source, torch.export.ExportedProgram):
            ep, name = source, "model"
        else:
            if example_input is None:
                raise ValueError("a PyTorch module needs example_input= (e.g. "
                                 "torch.randn(1, 3, 32, 32)) to be exported")
            ex = example_input if isinstance(example_input, tuple) else (example_input,)
            ex = tuple(torch.as_tensor(np.asarray(e, np.float32)) if not torch.is_tensor(e) else e
                       for e in ex)
            was_training = source.training
            source.eval()
            try:
                with torch.no_grad():
                    ep = torch.export.export(source, ex)
            finally:
                source.train(was_training)
            name = type(source).__name__.lower()
        return _Importer(ep, name, channels_last_io).run()


class _Importer(object):
    def __init__(self, ep, name, channels_last_io):
        import torch
        self.torch = torch
        self.ep = ep
        self.g = Graph(name)
        self.g.source = "pytorch"
        self.channels_last_io = channels_last_io
        self.env = {}
        self.cache = {}
        self.node_name = "?"
        self.grouped = 0

    # -- graph building helpers ----------------------------------------------

    def tensor(self, base, ir_shape):
        return self.g.add_tensor(base, ir_shape).name

    def add(self, op, inputs, ir_shape, attrs=None, base=None):
        y = self.tensor(base or self.node_name, ir_shape)
        self.g.add_node(op, inputs, [y], attrs, name=y)
        return y

    def const(self, value, base="c"):
        return self.g.add_const("%s_%s" % (self.node_name, base),
                                np.asarray(value, np.float32)).name

    def val(self, a):
        if isinstance(a, self.torch.fx.Node):
            v = self.env[a.name]
        elif isinstance(a, (list, tuple)):
            return type(a)(self.val(x) for x in a)
        else:
            v = a
        if isinstance(v, _Unavailable):
            raise NotImplementedError(v.why)
        return v

    def fail(self, what):
        raise NotImplementedError("PyTorch %s (node %s) has no pTensor mapping" % (what, self.node_name))

    # -- layouts ----------------------------------------------------------------

    def layout(self, v, perm):
        """v with its IR tensor in layout `perm` (torch dims in IR order; dims of
        size 1 may be left out). A reshape when only unit dims move, else a
        transpose."""
        perm = tuple(perm)
        if v.flat is not None:
            v = self.unflatten(v)
        if v.perm == perm:
            return v
        key = (v.name, perm)
        if key not in self.cache:
            shape_new = tuple(v.shape[p] for p in perm) or (1,)
            big_now = [p for p in v.perm if v.shape[p] != 1]
            big_new = [p for p in perm if v.shape[p] != 1]
            if sorted(big_now) != sorted(big_new):
                raise AssertionError("layout %s does not cover %s" % (perm, v.perm))
            if big_now == big_new:
                name = self.add("reshape", [v.name], shape_new, base=v.name + "_r")
            else:
                cur = list(v.perm) + [d for d in perm if d not in v.perm]
                src = v.name
                if len(cur) != len(v.perm):
                    src = self.add("reshape", [src], tuple(v.shape[d] for d in cur),
                                   base=v.name + "_r")
                extra = [d for d in cur if d not in perm]
                tgt = list(perm) + extra
                q = tuple(cur.index(d) for d in tgt)
                name = self.add("transpose", [src], tuple(v.shape[d] for d in tgt),
                                {"perm": q}, base=v.name + "_t")
                if extra:
                    name = self.add("reshape", [name], shape_new, base=v.name + "_r")
            self.cache[key] = name
        return _Value(self.cache[key], perm, v.shape)

    def ident(self, v):
        return self.layout(v, _ident(v.rank))

    def unflatten(self, v):
        """Materialize a pending flatten: features in PyTorch's C, H, W order."""
        src = self.env_flat[v.name]
        t = self.layout(src, _ident(4))  # NCHW
        y = self.add("reshape", [t.name], v.shape, base=v.name + "_chw")
        return _Value(y, _ident(v.rank), v.shape)

    def free_order(self, v):
        """True when the IR tensor's elements are in PyTorch's order."""
        if v.flat is not None:
            return False
        big = [p for p in v.perm if v.shape[p] != 1]
        return big == sorted(big)

    def reshape(self, v, shape):
        shape = tuple(int(d) for d in shape)
        if not self.free_order(v):
            v = self.ident(v)
        perm = _ident(len(shape))
        if len(shape) > 4:  # keep only the dims that are not 1 in the IR
            perm = tuple(d for d in perm if shape[d] != 1)
            if len(perm) > 4:
                self.fail("a rank-%d tensor (pTensor has at most 4 dims)" % len(perm))
        ir = tuple(shape[d] for d in perm) or (1,)
        if v.ir_shape == ir:
            return _Value(v.name, perm, shape)
        y = self.add("reshape", [v.name], ir)
        return _Value(y, perm, shape)

    def nhwc(self, v):
        if v.rank != 4:
            self.fail("operator on a rank-%d tensor (needs NCHW)" % v.rank)
        return self.layout(v, NHWC)

    def channel_last(self, v):
        """Layout with PyTorch dim 1 (channels) last: NHWC, (N, L, C), (N, C)."""
        if v.rank == 2:
            return self.ident(v)
        p = (0,) + tuple(range(2, v.rank)) + (1,)
        return self.layout(v, p)

    def const_like(self, arr, v):
        """A constant shaped to broadcast with v's IR tensor."""
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > v.rank:
            self.fail("broadcast of a rank-%d constant onto a rank-%d tensor" % (arr.ndim, v.rank))
        if v.flat is not None:
            self.fail("broadcast onto a flattened tensor")
        full = arr.reshape((1,) * (v.rank - arr.ndim) + arr.shape)
        out = self.const_layout(full, v.perm)
        while out.ndim > 1 and out.shape[0] == 1:
            out = out[0]
        return self.g.add_const(self.node_name + "_k", np.ascontiguousarray(out)).name

    @staticmethod
    def const_layout(full, perm):
        """A constant with torch shape -> IR layout `perm` (left-out dims are 1)."""
        omitted = [d for d in range(full.ndim) if d not in perm]
        out = np.transpose(full, tuple(perm) + tuple(omitted))
        return out.reshape(out.shape[:len(perm)] or (1,))

    # -- the walk -------------------------------------------------------------

    def run(self):
        ep, g = self.ep, self.g
        sig = ep.graph_signature
        consts = {}
        for spec in sig.input_specs:
            kind = spec.kind.name
            name = spec.arg.name
            if kind == "PARAMETER" or kind == "BUFFER":
                t = ep.state_dict[spec.target] if spec.target in ep.state_dict else \
                    ep.constants[spec.target]
                consts[name] = t
            elif kind in ("CONSTANT_TENSOR", "CUSTOM_OBJ"):
                consts[name] = ep.constants[spec.target]
        self.env_flat = {}
        nodes = list(ep.graph.nodes)
        for node in nodes:
            self.node_name = node.name
            if node.op == "placeholder":
                if node.name in consts:
                    self.env[node.name] = consts[node.name].detach().cpu().numpy()
                    continue
                shape = tuple(int(d) for d in node.meta["val"].shape)
                if len(shape) == 4 and self.channels_last_io:
                    perm = NHWC
                else:
                    perm = _ident(len(shape))
                name = g.add_tensor(node.name, tuple(shape[p] for p in perm)).name
                g.inputs.append(name)
                self.env[node.name] = _Value(name, perm, shape)
            elif node.op == "call_function":
                self.env[node.name] = self.call(node)
            elif node.op == "output":
                for o in node.args[0]:
                    v = self.val(o)
                    if not isinstance(v, _Value):
                        raise NotImplementedError("model output %s is a constant" % o)
                    if v.rank == 4 and self.channels_last_io:
                        v = self.layout(v, NHWC)
                    else:
                        v = self.ident(v)
                    name = v.name
                    if name in g.inputs or name in g.outputs:
                        name = self.add("reshape", [name], v.ir_shape, base=name + "_out")
                    g.outputs.append(name)
            else:
                raise NotImplementedError("fx node kind %s" % node.op)
        g.remove_unused()
        if self.grouped:
            g.notes.append("pytorch: %d grouped convolution(s) run as dense convolutions with "
                           "block-diagonal filters" % self.grouped)
        return g

    def call(self, node):
        target = node.target
        if target is operator.getitem:
            seq = self.val(node.args[0])
            return seq[node.args[1]]
        packet = getattr(target, "_overloadpacket", None)
        op = packet.__name__ if packet is not None else getattr(target, "__name__", str(target))
        op = op.rstrip("_") if op not in ("_softmax", "_log_softmax", "_unsafe_view", "_to_copy",
                                          "_prelu_kernel", "_adaptive_avg_pool2d",
                                          "_native_batch_norm_legit_no_training",
                                          "_assert_tensor_metadata", "_upsample_nearest_exact2d",
                                          "_upsample_nearest_exact1d") else op
        args = [self.val(a) for a in node.args]
        kwargs = {k: self.val(v) for k, v in node.kwargs.items()}

        def has_value(x):
            if isinstance(x, _Value):
                return True
            if isinstance(x, (list, tuple)):
                return any(has_value(i) for i in x)
            return False

        if op == "_assert_tensor_metadata":
            return None
        if not has_value(args) and not has_value(list(kwargs.values())):
            return self.fold(target, args, kwargs)
        out_meta = node.meta.get("val")
        handler = getattr(self, "op_" + op, None)
        if handler is not None:
            return handler(node, args, kwargs, out_meta)
        if op in _IDENTITY:
            return args[0]
        if op in _UNARY:
            return self.unary(args[0], _UNARY[op])
        if op in _BINARY:
            return self.binary(_BINARY[op], args[0], args[1], kwargs.get("alpha", 1)
                               if len(args) < 3 else args[2])
        self.fail("operator aten.%s" % op)

    def fold(self, target, args, kwargs):
        torch = self.torch

        def to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(np.ascontiguousarray(x))
            if isinstance(x, (list, tuple)):
                return type(x)(to_t(i) for i in x)
            return x

        def to_np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            if isinstance(x, (list, tuple)):
                return type(x)(to_np(i) for i in x)
            return x

        with torch.no_grad():
            return to_np(target(*to_t(args), **to_t(kwargs)))

    # -- elementwise ------------------------------------------------------------

    def unary(self, v, kind, **attrs):
        if v.flat is not None:  # flatten(f(x)) == f(flatten(x)): apply f before the flatten
            src = self.unary(self.env_flat[v.name], kind, **attrs)
            return self.pending_flatten(src, v.shape)
        y = self.add("activation", [v.name], v.ir_shape, dict(kind=kind, **attrs))
        return _Value(y, v.perm, v.shape)

    def binary(self, kind, a, b, alpha=1):
        if kind in ("add", "sub") and alpha != 1:
            b = self.binary("mul", b, alpha) if isinstance(b, _Value) else np.asarray(b) * alpha
        if not isinstance(a, _Value):
            if kind == "add":
                return self.binary("add", b, a)
            if kind in ("mul", "maximum", "minimum"):
                return self.binary(kind, b, a)
        va = a if isinstance(a, _Value) else None
        vb = b if isinstance(b, _Value) else None
        if va is not None and vb is not None:
            if va.rank != vb.rank:  # right-align: leading unit dims on the smaller one
                if va.rank < vb.rank:
                    va = self.unsqueeze_leading(va, vb.rank)
                else:
                    vb = self.unsqueeze_leading(vb, va.rank)
            if va.flat is not None or vb.flat is not None:
                va, vb = self.ident(va), self.ident(vb)
            shape = tuple(max(p, q) for p, q in zip(va.shape, vb.shape))
            bigs = [d for d in range(len(shape)) if shape[d] != 1]
            if all(d in va.perm for d in bigs):
                ref = va
            elif all(d in vb.perm for d in bigs):
                ref = vb
            else:
                ref = self.layout(va, va.perm + tuple(d for d in bigs if d not in va.perm))
            va, vb = self.layout(va, ref.perm), self.layout(vb, ref.perm)
            ia, ib = va.name, vb.name
        else:
            ref = va or vb
            if ref.flat is not None:
                ref = self.ident(ref)
            other = np.asarray(b if va is not None else a, np.float32)
            if other.ndim > ref.rank:
                self.fail("broadcast of a rank-%d constant" % other.ndim)
            shape = np.broadcast_shapes(ref.shape, other.shape)
            if tuple(shape) != ref.shape:
                self.fail("broadcast that grows a tensor")
            k = self.const_like(other, ref)
            if kind == "pow" and va is not None and other.size == 1:
                e = float(other.reshape(-1)[0])
                if e == 2.0:
                    return self.unary(ref, "square")
                if e == 0.5:
                    return self.unary(ref, "sqrt")
                if e == -0.5:
                    return self.unary(ref, "rsqrt")
                if e == 1.0:
                    return ref
            ia, ib = (ref.name, k) if va is not None else (k, ref.name)
        ir_shape = tuple(shape[p] for p in ref.perm)
        if kind == "add":
            y = self.add("add", [ia, ib], ir_shape, {"activation": None})
        else:
            y = self.add("binary", [ia, ib], ir_shape, {"kind": kind})
        return _Value(y, ref.perm, shape)

    def op_max(self, node, args, kwargs, meta, kind="maximum"):
        if len(args) == 2 and not isinstance(args[1], int):
            return self.binary(kind, args[0], args[1])
        if len(args) >= 2 and isinstance(args[1], int):  # max.dim -> (values, indices)
            keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            vals = self.reduce_max(args[0], [args[1]], keep, kind == "minimum")
            return (vals, _Unavailable("%s indices" % kind[:3]))
        self.fail("max / min over the whole tensor")

    def op_min(self, node, args, kwargs, meta):
        return self.op_max(node, args, kwargs, meta, kind="minimum")

    def op_amax(self, node, args, kwargs, meta, neg=False):
        dims = args[1] if len(args) > 1 else kwargs.get("dim", [])
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return self.reduce_max(args[0], dims, keep, neg)

    def op_amin(self, node, args, kwargs, meta):
        return self.op_amax(node, args, kwargs, meta, neg=True)

    def reduce_max(self, x, dims, keep, neg=False):
        """Max (or min) over the spatial dims of an N, C, H, W / N, C, L tensor,
        run as a max pool over the whole image."""
        dims = sorted(set(d % x.rank for d in (dims if isinstance(dims, (list, tuple)) else [dims])))
        if neg:
            return self.unary(self.reduce_max(self.unary(x, "neg"), dims, keep), "neg")
        if not ((x.rank == 4 and dims and set(dims) <= {2, 3}) or (x.rank == 3 and dims == [2])):
            self.fail("max / min over dims %s of a rank-%d tensor (only over H, W)" % (dims, x.rank))
        x4, back = self.to_nhwc(x)
        _, H, W, C = x4.ir_shape
        kh = H if (x.rank == 4 and 2 in dims) else 1
        kw = W if (3 in dims or x.rank == 3) else 1
        oh, ow = H // kh, W // kw
        y = self.add("max_pool2d", [x4.name], (1, oh, ow, C),
                     {"kernel": (kh, kw), "stride": (kh, kw), "padding": "VALID"})
        v = back(y, C, oh, ow)
        return v if keep else self.drop_dims(v, dims)

    def op_var(self, node, args, kwargs, meta, sqrt=False):
        x = args[0]
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        if "correction" in kwargs or (len(args) > 2 and not isinstance(args[2], bool)):
            corr = kwargs.get("correction", args[2] if len(args) > 2 else 1)
            corr = 1 if corr is None else corr
        else:
            unbiased = args[2] if len(args) > 2 else kwargs.get("unbiased", True)
            corr = 1 if unbiased else 0
        keep = kwargs.get("keepdim", args[3] if len(args) > 3 else False)
        if dims is None:
            dims = list(range(x.rank))
        dl = [d % x.rank for d in (dims if isinstance(dims, (list, tuple)) else [dims])]
        n = int(np.prod([x.shape[d] for d in dl]))
        m = self.reduce_mean(x, dl, True)
        d = self.binary("sub", x, m)
        v = self.reduce_mean(self.unary(d, "square"), dl, keep)
        if corr:
            v = self.binary("mul", v, np.float32(n / max(n - corr, 1)))
        return self.unary(v, "sqrt") if sqrt else v

    def op_std(self, node, args, kwargs, meta):
        return self.op_var(node, args, kwargs, meta, sqrt=True)

    def tile(self, x, reps):
        """np.tile: copies along each dim, as a chain of concats."""
        if len(reps) > x.rank:
            x = self.unsqueeze_leading(x, len(reps))
        reps = (1,) * (x.rank - len(reps)) + tuple(reps)
        for d, r in enumerate(reps):
            if r > 1:
                x = self.op_cat(None, [[x] * int(r), d], {}, None)
        return x

    def op_repeat(self, node, args, kwargs, meta):
        return self.tile(args[0], args[1])

    def op_expand(self, node, args, kwargs, meta):
        x, sizes = args[0], list(args[1])
        if len(sizes) > x.rank:
            x = self.unsqueeze_leading(x, len(sizes))
        reps = []
        for d, s_ in enumerate(sizes):
            if s_ == -1 or s_ == x.shape[d]:
                reps.append(1)
            elif x.shape[d] == 1:
                reps.append(int(s_))
            else:
                self.fail("expand of dim %d from %d to %d" % (d, x.shape[d], s_))
        return self.tile(x, reps)

    def op_upsample_nearest2d(self, node, args, kwargs, meta):
        x = args[0]
        out = self.out_shape(meta)
        x4, back = self.to_nhwc(x)
        _, H, W, C = x4.ir_shape
        oh, ow = (out[-2], out[-1]) if x.rank == 4 else (1, out[-1])
        if oh % H or ow % W:
            self.fail("nearest upsampling by a non-integer factor")
        sh, sw = oh // H, ow // W
        cur = x4.name
        if sw > 1:  # repeat each pixel: concat along C, then view as W * sw
            t = cur
            for _ in range(sw - 1):
                t = self.add("concat", [t, cur], (1, H, W, C * (_ + 2)), {"axis": 3},
                             base=self.node_name + "_w")
            cur = self.add("reshape", [t], (1, H, W * sw, C), base=self.node_name + "_wv")
        if sh > 1:  # repeat each row
            row = self.add("reshape", [cur], (1, H, 1, W * sw * C), base=self.node_name + "_row")
            t = row
            for i in range(sh - 1):
                t = self.add("concat", [t, row], (1, H, i + 2, W * sw * C), {"axis": 2},
                             base=self.node_name + "_h")
            cur = self.add("reshape", [t], (1, H * sh, W * sw, C), base=self.node_name + "_hv")
        return back(cur, C, H * sh, W * sw)

    op_upsample_nearest1d = op_upsample_nearest2d
    op__upsample_nearest_exact2d = op_upsample_nearest2d
    op__upsample_nearest_exact1d = op_upsample_nearest2d

    def unsqueeze_leading(self, v, rank):
        """v with unit dims added in front up to `rank` (free)."""
        k = rank - v.rank
        if v.flat is not None:
            v = self.ident(v)
        return _Value(v.name, tuple(p + k for p in v.perm), (1,) * k + v.shape)

    def op_rsub(self, node, args, kwargs, meta):
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        a, b = args[0], args[1]
        if alpha != 1:
            a = self.binary("mul", a, alpha)
        return self.binary("sub", b, a)

    def op_reciprocal(self, node, args, kwargs, meta):
        return self.binary("div", np.float32(1.0), args[0])

    def op_clamp(self, node, args, kwargs, meta):
        x = args[0]
        lo = kwargs.get("min", args[1] if len(args) > 1 else None)
        hi = kwargs.get("max", args[2] if len(args) > 2 else None)
        if isinstance(lo, _Value) or isinstance(hi, _Value) or \
                isinstance(lo, np.ndarray) or isinstance(hi, np.ndarray):
            if lo is not None:
                x = self.binary("maximum", x, lo)
            if hi is not None:
                x = self.binary("minimum", x, hi)
            return x
        return self.clip(x, lo, hi)

    op_clip = op_clamp

    def op_clamp_min(self, node, args, kwargs, meta):
        return self.clip(args[0], args[1], None)

    def op_clamp_max(self, node, args, kwargs, meta):
        return self.clip(args[0], None, args[1])

    def clip(self, x, lo, hi):
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        named = {(0.0, np.inf): "relu", (0.0, 6.0): "relu6", (-1.0, 1.0): "relu1"}
        if (lo, hi) in named:
            return self.unary(x, named[(lo, hi)])
        if np.isinf(lo) and np.isinf(hi):
            return x
        if np.isinf(lo):
            return self.binary("minimum", x, np.float32(hi))
        if np.isinf(hi):
            return self.binary("maximum", x, np.float32(lo))
        return self.unary(x, "clip", lo=lo, hi=hi)

    def op_hardtanh(self, node, args, kwargs, meta):
        lo = args[1] if len(args) > 1 else kwargs.get("min_val", -1.0)
        hi = args[2] if len(args) > 2 else kwargs.get("max_val", 1.0)
        return self.clip(args[0], lo, hi)

    def op_leaky_relu(self, node, args, kwargs, meta):
        return self.unary(args[0], "leaky_relu", alpha=float(args[1] if len(args) > 1 else
                                                             kwargs.get("negative_slope", 0.01)))

    def op_elu(self, node, args, kwargs, meta):
        alpha = float(args[1] if len(args) > 1 else kwargs.get("alpha", 1.0))
        scale = float(args[2] if len(args) > 2 else kwargs.get("scale", 1.0))
        in_scale = float(args[3] if len(args) > 3 else kwargs.get("input_scale", 1.0))
        x = args[0]
        if in_scale != 1.0:
            x = self.binary("mul", x, np.float32(in_scale))
        y = self.unary(x, "elu", alpha=alpha)
        if scale != 1.0:
            y = self.binary("mul", y, np.float32(scale))
        return y

    def op_celu(self, node, args, kwargs, meta):
        return self.unary(args[0], "celu", alpha=float(args[1] if len(args) > 1 else
                                                       kwargs.get("alpha", 1.0)))

    def op_gelu(self, node, args, kwargs, meta):
        approx = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        return self.unary(args[0], "gelu_tanh" if approx == "tanh" else "gelu")

    def op_softplus(self, node, args, kwargs, meta):
        beta = float(args[1] if len(args) > 1 else kwargs.get("beta", 1.0))
        threshold = float(args[2] if len(args) > 2 else kwargs.get("threshold", 20.0))
        if threshold != 20.0:
            self.fail("softplus with threshold %g" % threshold)
        x = args[0]
        if beta == 1.0:
            return self.unary(x, "softplus")
        y = self.unary(self.binary("mul", x, np.float32(beta)), "softplus")
        return self.binary("mul", y, np.float32(1.0 / beta))

    def op_prelu(self, node, args, kwargs, meta):
        # max(x, 0) + a * min(x, 0); a is [C] (dim 1) or a scalar
        x, a = args[0], np.asarray(args[1], np.float32).reshape(-1)
        if a.size == 1:
            if float(a[0]) >= 0:
                return self.unary(x, "leaky_relu", alpha=float(a[0]))
            alpha = a.reshape(1)
        else:
            alpha = a.reshape((a.size,) + (1,) * (x.rank - 2))
        neg = self.binary("mul", self.binary("minimum", x, np.float32(0.0)), alpha)
        return self.binary("add", self.unary(x, "relu"), neg)

    op__prelu_kernel = op_prelu

    # -- convolution and linear ---------------------------------------------------

    def op_conv1d(self, node, args, kwargs, meta):
        return self.conv(args, kwargs, meta, padding_str=None)

    def op_conv2d(self, node, args, kwargs, meta):
        return self.conv(args, kwargs, meta)

    def op_convolution(self, node, args, kwargs, meta):
        x, w, b, stride, padding, dilation, transposed, _, groups = args[:9]
        if transposed:
            self.fail("transposed convolution")
        return self.conv([x, w, b, stride, padding, dilation, groups], {}, meta)

    def conv(self, args, kwargs, meta, padding_str=None):
        names = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
        a = dict(zip(names, args))
        a.update(kwargs)
        x, w = a["input"], a["weight"]
        if isinstance(w, _Value):
            self.fail("convolution with a computed filter")
        w = np.asarray(w, np.float32)
        b = a.get("bias")
        groups = int(a.get("groups", 1))
        one_d = w.ndim == 3
        if one_d:
            w = w[:, :, None, :]
        O, Ig, kh, kw = w.shape
        n = 1 if one_d else 2
        stride = _pair(a.get("stride", 1), n)
        dil = _pair(a.get("dilation", 1), n)
        pad = a.get("padding", 0)
        if one_d:
            stride, dil = (1,) + stride, (1,) + dil
        x4, back = self.to_nhwc(x)
        H, W, C = x4.ir_shape[1], x4.ir_shape[2], x4.ir_shape[3]
        if isinstance(pad, str):
            pad_mode = pad.upper()
            pads = None
        else:
            pads = _pair(pad, n)
            if one_d:
                pads = (0,) + pads
            pad_mode = None
        oh, ow = int(meta.shape[-2]) if not one_d else 1, int(meta.shape[-1])
        padding, x4 = self.tf_padding(x4, (H, W), (kh, kw), stride, dil, pads, pad_mode, (oh, ow), 0.0)
        bias = np.zeros(O, np.float32) if b is None else np.asarray(b, np.float32)
        if isinstance(b, _Value):
            self.fail("convolution with a computed bias")
        attrs = {"stride": stride, "padding": padding, "dilation": dil, "activation": None}
        if groups == 1:
            f = self.const(np.transpose(w, (0, 2, 3, 1)), "filter")
            y = self.add("conv2d", [x4.name, f, self.const(bias, "bias")], (1, oh, ow, O), attrs)
        elif groups == C and Ig == 1 and O % C == 0:
            attrs["depth_multiplier"] = O // C
            f = self.const(np.transpose(w[:, 0], (1, 2, 0))[None], "filter")
            y = self.add("depthwise_conv2d", [x4.name, f, self.const(bias, "bias")],
                         (1, oh, ow, O), attrs)
        else:
            full = np.zeros((O, kh, kw, C), np.float32)
            og = O // groups
            for gi in range(groups):
                full[gi * og:(gi + 1) * og, :, :, gi * Ig:(gi + 1) * Ig] = \
                    np.transpose(w[gi * og:(gi + 1) * og], (0, 2, 3, 1))
            self.grouped += 1
            y = self.add("conv2d", [x4.name, self.const(full, "filter"), self.const(bias, "bias")],
                         (1, oh, ow, O), attrs)
        return back(y, O, oh, ow)

    def to_nhwc(self, x):
        """x as an NHWC value (rank 3 inputs are N, C, L -> N, 1, L, C) and a
        function that turns an NHWC result back into the matching layout."""
        if x.rank == 4:
            v = self.nhwc(x)
            return v, lambda y, c, h, w: _Value(y, NHWC, (1, c, h, w))
        if x.rank != 3:
            self.fail("convolution / pooling on a rank-%d tensor" % x.rank)
        v = self.layout(x, (0, 2, 1))
        n, L, c = v.ir_shape
        r = self.add("reshape", [v.name], (n, 1, L, c), base=v.name + "_2d")

        def back(y, c, h, w):
            z = self.add("reshape", [y], (1, w, c), base=y + "_1d")
            return _Value(z, (0, 2, 1), (1, c, w))
        return _Value(r, NHWC, (n, c, 1, L)), back

    def tf_padding(self, x4, size, k, stride, dil, pads, pad_mode, out, fill, avg=None):
        """Express PyTorch's padding as TF SAME / VALID, inserting a pad node
        when neither matches. Returns (padding, input value)."""
        if pad_mode == "VALID":
            pads = (0, 0)
        if pad_mode == "SAME":  # PyTorch pads (k_eff - 1) // 2 before, the rest after: TF SAME
            if stride != (1, 1):
                self.fail("padding='same' with stride %s" % (stride,))
            return "SAME", x4
        if pads == (0, 0) and all(conv_out_size(size[i], k[i], stride[i], "VALID", dil[i]) == out[i]
                                  for i in range(2)):
            return "VALID", x4
        same = True
        for i in range(2):
            eff = (k[i] - 1) * dil[i] + 1
            total = max(0, (out[i] - 1) * stride[i] + eff - size[i])
            if conv_out_size(size[i], k[i], stride[i], "SAME", dil[i]) != out[i] or \
                    pads[i] != total // 2:
                same = False
        if same and avg in (None, False):
            return "SAME", x4
        # explicit padding: pad, then VALID with the PyTorch output size
        after = []
        for i in range(2):
            eff = (k[i] - 1) * dil[i] + 1
            need = (out[i] - 1) * stride[i] + eff - size[i] - pads[i]
            after.append(max(0, need))
            if avg is True and need > pads[i]:
                self.fail("average pooling with ceil_mode windows past the padding")
        if pads == (0, 0) and after == [0, 0]:
            return "VALID", x4
        if avg is False:
            self.fail("average pooling with count_include_pad=False and padding / ceil_mode "
                      "that TF SAME does not reproduce")
        return "VALID", self.pad_node(x4, (pads[0], after[0], pads[1], after[1]), fill)

    def pad_node(self, x4, pads, fill):
        n, h, w, c = x4.ir_shape
        t, b, l, r = (int(p) for p in pads)
        y = self.add("pad", [x4.name], (n, h + t + b, w + l + r, c),
                     {"pads": (t, b, l, r), "value": float(fill)}, base=x4.name + "_pad")
        return _Value(y, NHWC, (n, c, h + t + b, w + l + r))

    def op_linear(self, node, args, kwargs, meta):
        x, w = args[0], args[1]
        b = args[2] if len(args) > 2 else kwargs.get("bias")
        if isinstance(w, _Value) or isinstance(b, _Value):
            y = self.matmul(x, self.op_t(None, [w], {}, None) if isinstance(w, _Value)
                            else np.asarray(w).T)
            return y if b is None else self.binary("add", y, b)
        return self.dense(x, np.asarray(w, np.float32).T, b, meta)

    def dense(self, x, wt, b, meta):
        """y = x @ wt + b over the last dim, wt constant [in, out]."""
        out_f = wt.shape[1]
        bias = np.zeros(out_f, np.float32) if b is None else np.asarray(b, np.float32)
        out_shape = tuple(int(d) for d in meta.shape)
        if x.flat is not None:
            C, H, W = x.flat
            idx = np.arange(C * H * W).reshape(C, H, W).transpose(1, 2, 0).reshape(-1)
            wt = wt[idx]
            xn = x.name
        else:
            if int(np.prod(x.shape[:-1])) != 1:
                # rows of a sequence: matmul + bias
                y = self.matmul(x, wt)
                return self.binary("add", y, bias) if b is not None else y
            xn = self.reshape(x, (1, x.shape[-1])).name
        y = self.add("dense", [xn, self.const(wt, "weights"), self.const(bias, "bias")],
                     (1, out_f), {"activation": None})
        v = _Value(y, (0, 1), (1, out_f))
        return v if out_shape == (1, out_f) else self.reshape(v, out_shape)

    def matmul(self, a, b):
        """a @ b with NumPy batching: a [..., M, K] (an activation), b [..., K, N]
        (an activation or a constant). Batches run one 2D matmul each."""
        if a.flat is not None:
            a = self.ident(a)
        if a.rank == 1:
            y = self.matmul(self.unsqueeze_leading(a, 2), b)
            return self.drop_dims(y, [y.rank - 2])
        bconst = not isinstance(b, _Value)
        bshape = np.shape(b) if bconst else b.shape
        if len(bshape) == 1:
            bb = np.asarray(b, np.float32)[:, None] if bconst else \
                _Value(b.name, b.perm, b.shape + (1,))
            y = self.matmul(a, bb)
            return self.drop_dims(y, [y.rank - 1])
        M, K = a.shape[-2:]
        N = bshape[-1]
        ab, bbat = a.shape[:-2], tuple(bshape[:-2])
        rank = max(len(ab), len(bbat))
        ab = (1,) * (rank - len(ab)) + ab
        bbat = (1,) * (rank - len(bbat)) + bbat
        batch = tuple(max(p, q) for p, q in zip(ab, bbat))
        out_shape = batch + (M, N)
        na, nb, nt = int(np.prod(ab)), int(np.prod(bbat)), int(np.prod(batch))
        if (na not in (1, nt)) or (nb not in (1, nt)):
            self.fail("matmul with partly broadcast batch dims %s @ %s" % (a.shape, bshape))
        if nb == 1:  # one right-hand side: fold a's batch into its rows, one matmul
            a2 = self.ident(self.reshape(a, (na * M, K)))
            if bconst:
                bn = self.const(np.asarray(b, np.float32).reshape(K, N), "rhs")
            else:
                bn = self.ident(self.reshape(b, (K, N))).name
            y = self.add("matmul", [a2.name, bn], (na * M, N))
            return self.reshape(_Value(y, (0, 1), (na * M, N)), out_shape)
        a3 = self.reshape(a, (na, M, K))
        if bconst:
            b3 = np.asarray(b, np.float32).reshape(nb, K, N)
        else:
            b3 = self.reshape(b, (nb, K, N))
        parts = []
        for i in range(nt):
            ai = self.drop_dims(self.slice_value(a3, 0, i if na > 1 else 0,
                                                 (i if na > 1 else 0) + 1), [0])
            ai = self.ident(ai)
            if bconst:
                bn = self.const(b3[i if nb > 1 else 0], "rhs")
            else:
                bi = self.drop_dims(self.slice_value(b3, 0, i if nb > 1 else 0,
                                                     (i if nb > 1 else 0) + 1), [0])
                bn = self.ident(bi).name
            y = self.add("matmul", [ai.name, bn], (M, N))
            parts.append(_Value(y, (0, 1), (M, N)))
        if nt == 1:
            return self.reshape(parts[0], out_shape)
        stacked = self.op_cat(None, [[self.reshape(p, (1, M, N)) for p in parts], 0], {}, None)
        return self.reshape(stacked, out_shape)

    def op_matmul(self, node, args, kwargs, meta):
        a, b = args[0], args[1]
        if not isinstance(a, _Value):  # W @ x = (x^T @ W^T)^T
            if b.rank != 2:
                self.fail("matmul of a constant with a rank-%d tensor" % b.rank)
            at = self.op_t(None, [b], {}, None)
            return self.op_t(None, [self.matmul(at, np.asarray(a, np.float32).T)], {}, None)
        if not isinstance(b, _Value) and np.asarray(b).ndim == 2:
            return self.dense(a, np.asarray(b, np.float32), None, meta)
        return self.matmul(a, b)

    op_mm = op_matmul
    op_bmm = op_matmul

    def op_scaled_dot_product_attention(self, node, args, kwargs, meta):
        names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale")
        a = dict(zip(names, args))
        a.update(kwargs)
        q, k, v = a["query"], a["key"], a["value"]
        mask = a.get("attn_mask")
        if a.get("dropout_p", 0.0):
            self.fail("attention dropout")
        if a.get("enable_gqa"):
            self.fail("grouped-query attention")
        D = q.shape[-1]
        scale = a.get("scale")
        scale = 1.0 / np.sqrt(D) if scale is None else float(scale)
        kt = self.op_transpose(None, [k, -2, -1], {}, None)
        scores = self.binary("mul", self.matmul(q, kt), np.float32(scale))
        L, S = q.shape[-2], k.shape[-2]
        if a.get("is_causal"):
            mask = np.tril(np.ones((L, S), bool))
        if mask is not None:
            if isinstance(mask, _Value):
                self.fail("attention with a computed mask")
            mask = np.asarray(mask)
            add = np.where(mask, 0.0, -1e30) if mask.dtype == bool else mask  # finite: flash data
            scores = self.binary("add", scores, add.astype(np.float32))
        probs = self.op_softmax(None, [scores, -1], {}, None)
        return self.matmul(probs, v)

    def op_addmm(self, node, args, kwargs, meta):
        bias, x, w = args[0], args[1], args[2]
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
        if beta != 1 or alpha != 1 or isinstance(w, _Value) or isinstance(bias, _Value):
            y = self.matmul(x, w)
            return self.binary("add", self.binary("mul", y, alpha) if alpha != 1 else y,
                               np.asarray(bias) * beta if not isinstance(bias, _Value) else bias)
        return self.dense(x, np.asarray(w, np.float32), np.asarray(bias, np.float32).reshape(-1),
                          meta)

    # -- normalization -------------------------------------------------------------

    def op_batch_norm(self, node, args, kwargs, meta):
        x, w, b, mean, var = args[:5]
        eps = float(args[7] if len(args) > 7 else kwargs.get("eps", 1e-5))
        if len(args) > 5 and args[5]:
            self.fail("batch_norm in training mode (call model.eval())")
        return self.bn(x, w, b, mean, var, eps)

    def op__native_batch_norm_legit_no_training(self, node, args, kwargs, meta):
        x, w, b, mean, var = args[:5]
        eps = float(args[6] if len(args) > 6 else 1e-5)
        y = self.bn(x, w, b, mean, var, eps)
        return (y, _Unavailable("batch norm saved mean"), _Unavailable("batch norm saved std"))

    def bn(self, x, w, b, mean, var, eps):
        v = self.channel_last(x)
        c = v.ir_shape[-1]
        w = np.ones(c, np.float32) if w is None else np.asarray(w, np.float32)
        b = np.zeros(c, np.float32) if b is None else np.asarray(b, np.float32)
        y = self.add("batch_norm", [v.name, self.const(mean, "mean"), self.const(var, "var"),
                                    self.const(w, "gamma"), self.const(b, "beta")],
                     v.ir_shape, {"epsilon": eps})
        return _Value(y, v.perm, v.shape)

    def normalize(self, v, axes, eps):
        """(v - mean) / sqrt(var + eps) over IR axes `axes` of value v."""
        keep = tuple(1 if i in axes else d for i, d in enumerate(v.ir_shape))
        torch_keep = tuple(keep[v.perm.index(i)] for i in range(v.rank))
        m = self.add("mean", [v.name], keep, {"axes": tuple(axes)}, base=self.node_name + "_mean")
        mv = _Value(m, v.perm, torch_keep)
        d = self.binary("sub", v, mv)
        sq = self.unary(d, "square")
        var = self.add("mean", [sq.name], keep, {"axes": tuple(axes)}, base=self.node_name + "_var")
        varv = _Value(var, v.perm, torch_keep)
        r = self.unary(self.binary("add", varv, np.float32(eps)), "rsqrt")
        return self.binary("mul", d, r)

    def op_layer_norm(self, node, args, kwargs, meta):
        x, shape = args[0], args[1]
        w = args[2] if len(args) > 2 else kwargs.get("weight")
        b = args[3] if len(args) > 3 else kwargs.get("bias")
        eps = float(args[4] if len(args) > 4 else kwargs.get("eps", 1e-5))
        k = len(shape)
        if x.flat is not None:
            x = self.ident(x)
        dims = set(range(x.rank - k, x.rank))
        axes = [i for i, p in enumerate(x.perm) if p in dims]
        y = self.normalize(x, axes, eps)
        if w is not None:
            y = self.binary("mul", y, np.asarray(w, np.float32))
        if b is not None:
            y = self.binary("add", y, np.asarray(b, np.float32))
        return y

    def op_native_layer_norm(self, node, args, kwargs, meta):
        return (self.op_layer_norm(node, args[:5], {}, meta[0]), _Unavailable("layer norm mean"),
                _Unavailable("layer norm rstd"))

    def op_group_norm(self, node, args, kwargs, meta):
        x, groups = args[0], int(args[1])
        w = args[2] if len(args) > 2 else kwargs.get("weight")
        b = args[3] if len(args) > 3 else kwargs.get("bias")
        eps = float(args[4] if len(args) > 4 else kwargs.get("eps", 1e-5))
        return self.group_norm(x, groups, w, b, eps)

    def op_native_group_norm(self, node, args, kwargs, meta):
        x, w, b, _, C, _, groups, eps = args[:8]
        return (self.group_norm(x, int(groups), w, b, float(eps)),
                _Unavailable("group norm mean"), _Unavailable("group norm rstd"))

    def group_norm(self, x, groups, w, b, eps):
        v = self.channel_last(x)
        shape = v.ir_shape
        C = shape[-1]
        spatial = int(np.prod(shape[1:-1]))
        r = self.add("reshape", [v.name], (1, spatial, groups, C // groups), base=self.node_name + "_g")
        rv = _Value(r, (0, 1, 2, 3), (1, spatial, groups, C // groups))
        y = self.normalize(rv, (1, 3), eps)
        back = self.add("reshape", [y.name], shape, base=self.node_name + "_ug")
        out = _Value(back, v.perm, v.shape)
        cshape = (C,) + (1,) * (x.rank - 2)
        if w is not None:
            out = self.binary("mul", out, np.asarray(w, np.float32).reshape(cshape))
        if b is not None:
            out = self.binary("add", out, np.asarray(b, np.float32).reshape(cshape))
        return out

    def op_instance_norm(self, node, args, kwargs, meta):
        x, w, b = args[0], args[1], args[2]
        use_input_stats = args[5] if len(args) > 5 else kwargs.get("use_input_stats", True)
        eps = float(args[7] if len(args) > 7 else kwargs.get("eps", 1e-5))
        if not use_input_stats:
            return self.bn(x, w, b, args[3], args[4], eps)
        return self.group_norm(x, x.shape[1], w, b, eps)

    # -- pooling ---------------------------------------------------------------------

    def pool(self, kind, x, k, stride, pads, ceil_mode, meta, count_include_pad=True,
             divisor=None, dilation=1, one_d=False):
        n = 1 if one_d else 2
        k = _pair(k, n)
        stride = _pair(stride, n) if stride not in (None, [], ()) else k
        pads = _pair(pads, n)
        if _pair(dilation, n) != (1,) * n:
            self.fail("dilated pooling")
        if divisor is not None:
            self.fail("avg_pool2d with divisor_override")
        if one_d:
            k, stride, pads = (1,) + k, (1,) + stride, (0,) + pads
        x4, back = self.to_nhwc(x)
        H, W, C = x4.ir_shape[1], x4.ir_shape[2], x4.ir_shape[3]
        shape = meta.shape if not isinstance(meta, (list, tuple)) else meta[0].shape
        oh, ow = (1 if one_d else int(shape[-2])), int(shape[-1])
        if kind == "max":
            padding, x4 = self.tf_padding(x4, (H, W), k, stride, (1, 1), pads, None, (oh, ow),
                                          -np.inf)
        elif count_include_pad and pads != (0, 0):
            padding, x4 = self.tf_padding(x4, (H, W), k, stride, (1, 1), pads, None, (oh, ow),
                                          0.0, avg=True)
        else:
            padding, x4 = self.tf_padding(x4, (H, W), k, stride, (1, 1), pads, None, (oh, ow),
                                          0.0, avg=False if pads != (0, 0) else True)
        y = self.add("max_pool2d" if kind == "max" else "avg_pool2d", [x4.name], (1, oh, ow, C),
                     {"kernel": k, "stride": stride, "padding": padding})
        return back(y, C, oh, ow)

    def op_max_pool2d(self, node, args, kwargs, meta):
        a = args + [None] * (6 - len(args))
        return self.pool("max", a[0], a[1], a[2] or kwargs.get("stride"),
                         a[3] if a[3] is not None else kwargs.get("padding", 0),
                         a[5] if a[5] is not None else kwargs.get("ceil_mode", False), meta,
                         dilation=a[4] if a[4] is not None else kwargs.get("dilation", 1))

    def op_max_pool1d(self, node, args, kwargs, meta):
        a = args + [None] * (6 - len(args))
        return self.pool("max", a[0], a[1], a[2] or kwargs.get("stride"),
                         a[3] if a[3] is not None else kwargs.get("padding", 0),
                         a[5] if a[5] is not None else kwargs.get("ceil_mode", False), meta,
                         dilation=a[4] if a[4] is not None else 1, one_d=True)

    def op_max_pool2d_with_indices(self, node, args, kwargs, meta):
        return (self.op_max_pool2d(node, args, kwargs, meta), _Unavailable("max pool indices"))

    def op_avg_pool2d(self, node, args, kwargs, meta, one_d=False):
        a = args + [None] * (7 - len(args))
        cip = a[5] if a[5] is not None else kwargs.get("count_include_pad", True)
        return self.pool("avg", a[0], a[1], a[2] or kwargs.get("stride"),
                         a[3] if a[3] is not None else kwargs.get("padding", 0),
                         a[4] if a[4] is not None else kwargs.get("ceil_mode", False), meta,
                         count_include_pad=cip,
                         divisor=a[6] if not one_d else None, one_d=one_d)

    def op_avg_pool1d(self, node, args, kwargs, meta):
        return self.op_avg_pool2d(node, args, kwargs, meta, one_d=True)

    def adaptive(self, kind, x, out, meta, one_d=False):
        out = _pair(out, 1 if one_d else 2)
        if one_d:
            out = (1,) + out
        x4, back = self.to_nhwc(x)
        H, W, C = x4.ir_shape[1], x4.ir_shape[2], x4.ir_shape[3]
        oh = H if out[0] is None else out[0]
        ow = W if out[1] is None else out[1]
        if H % oh or W % ow:
            self.fail("adaptive pooling from %dx%d to %dx%d (sizes must divide)" % (H, W, oh, ow))
        if kind == "avg" and (oh, ow) == (1, 1) and not one_d:
            y = self.add("global_avg_pool", [x4.name], (1, 1, 1, C))
            return back(y, C, 1, 1)
        k = (H // oh, W // ow)
        y = self.add("max_pool2d" if kind == "max" else "avg_pool2d", [x4.name], (1, oh, ow, C),
                     {"kernel": k, "stride": k, "padding": "VALID"})
        return back(y, C, oh, ow)

    def op_adaptive_avg_pool2d(self, node, args, kwargs, meta):
        return self.adaptive("avg", args[0], args[1], meta)

    op__adaptive_avg_pool2d = op_adaptive_avg_pool2d

    def op_adaptive_avg_pool1d(self, node, args, kwargs, meta):
        return self.adaptive("avg", args[0], args[1], meta, one_d=True)

    def op_adaptive_max_pool2d(self, node, args, kwargs, meta):
        return (self.adaptive("max", args[0], args[1], meta), _Unavailable("max pool indices"))

    def op_adaptive_max_pool1d(self, node, args, kwargs, meta):
        return (self.adaptive("max", args[0], args[1], meta, one_d=True),
                _Unavailable("max pool indices"))

    # -- shapes ------------------------------------------------------------------------

    def out_shape(self, meta):
        return tuple(int(d) for d in meta.shape)

    def op_flatten(self, node, args, kwargs, meta):
        x = args[0]
        shape = self.out_shape(meta)
        start = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        start = start + x.rank if start < 0 else start
        if x.rank == 4 and x.perm == NHWC and start == 1 and len(shape) == 2 \
                and not self.free_order(x):
            return self.pending_flatten(x, shape)
        return self.reshape(x, shape)

    def pending_flatten(self, x, shape):
        """Flatten an NHWC value without reordering: the features stay in H, W, C
        order until something needs PyTorch's C, H, W order."""
        y = self.add("reshape", [x.name], shape, base=self.node_name)
        self.env_flat[y] = x
        return _Value(y, _ident(len(shape)), shape, flat=x.shape[1:])

    def op_view(self, node, args, kwargs, meta):
        return self.reshape(args[0], self.out_shape(meta))

    op_reshape = op_view
    op__unsafe_view = op_view
    op_unflatten = op_view

    def op_squeeze(self, node, args, kwargs, meta):
        x = args[0]
        if len(args) > 1:
            dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
            dims = [d % x.rank for d in dims if x.shape[d % x.rank] == 1]
        else:
            dims = [d for d in range(x.rank) if x.shape[d] == 1]
        return self.drop_dims(x, dims)

    def drop_dims(self, x, dims):
        """Remove size-1 torch dims `dims` (free: the IR drops the same axes)."""
        if not dims:
            return x
        if x.flat is not None:
            x = self.ident(x)
        keep = [d for d in range(x.rank) if d not in dims]
        shape = tuple(x.shape[d] for d in keep)
        perm = tuple(keep.index(p) for p in x.perm if p in keep)
        if not any(d in x.perm for d in dims):  # those dims are not in the IR: free
            return _Value(x.name, perm, shape)
        ir_shape = tuple(shape[p] for p in perm) or (1,)
        y = self.add("reshape", [x.name], ir_shape)
        return _Value(y, perm, shape)

    def op_unsqueeze(self, node, args, kwargs, meta):
        x = args[0]
        if x.flat is not None:
            x = self.ident(x)
        d = args[1] % (x.rank + 1)
        shape = x.shape[:d] + (1,) + x.shape[d:]
        return _Value(x.name, tuple(p if p < d else p + 1 for p in x.perm), shape)

    def op_permute(self, node, args, kwargs, meta):
        x, q = args[0], [d % args[0].rank for d in args[1]]
        if x.flat is not None:
            x = self.ident(x)
        invq = [0] * len(q)
        for i, v in enumerate(q):
            invq[v] = i
        perm = tuple(invq[p] for p in x.perm)
        return _Value(x.name, perm, tuple(x.shape[d] for d in q))

    def op_transpose(self, node, args, kwargs, meta):
        x = args[0]
        d0, d1 = args[1] % x.rank, args[2] % x.rank
        q = list(range(x.rank))
        q[d0], q[d1] = q[d1], q[d0]
        return self.op_permute(node, [x, q], {}, meta)

    def op_t(self, node, args, kwargs, meta):
        x = args[0]
        if x.rank < 2:
            return x
        return self.op_permute(node, [x, [1, 0]], {}, meta)

    op_numpy_T = op_t

    def op_cat(self, node, args, kwargs, meta):
        items = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        vals = [v for v in items if isinstance(v, _Value)]
        ref = vals[0]
        if ref.flat is not None:
            ref = self.ident(ref)
        dim = dim % ref.rank
        if dim not in ref.perm:
            ref = self.layout(ref, ref.perm + (dim,))
        parts = []
        for v in items:
            if isinstance(v, _Value):
                parts.append(self.layout(v, ref.perm))
            else:
                arr = np.asarray(v, np.float32)
                k = self.g.add_const(self.node_name + "_k",
                                     np.ascontiguousarray(self.const_layout(arr, ref.perm))).name
                parts.append(_Value(k, ref.perm, arr.shape))
        axis = ref.perm.index(dim)
        cur = parts[0]
        for p in parts[1:]:
            shape = list(cur.shape)
            shape[dim] += p.shape[dim]
            shape = tuple(shape)
            y = self.add("concat", [cur.name, p.name], tuple(shape[i] for i in ref.perm),
                         {"axis": axis})
            cur = _Value(y, ref.perm, shape)
        return cur

    op_concat = op_cat

    def op_stack(self, node, args, kwargs, meta):
        items = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        rank = items[0].rank + 1 if isinstance(items[0], _Value) else np.asarray(items[0]).ndim + 1
        dim = dim % rank
        ups = []
        for v in items:
            if isinstance(v, _Value):
                shape = v.shape[:dim] + (1,) + v.shape[dim:]
                ups.append(self.reshape(v, shape))
            else:
                ups.append(np.expand_dims(np.asarray(v, np.float32), dim))
        return self.op_cat(node, [ups, dim], {}, meta)

    def slice_value(self, x, dim, start, stop, step=1):
        """x[..., start:stop:step, ...] along torch dim `dim`."""
        if x.flat is not None:
            x = self.ident(x)
        size = x.shape[dim]
        start, stop, step = slice(start, stop, step).indices(size)
        count = max(0, (stop - start + step - 1) // step)
        if count == size and step == 1:
            return x
        if count == 0:
            self.fail("empty slice")
        if dim not in x.perm:
            x = self.layout(x, x.perm + (dim,))
        shape = list(x.shape)
        shape[dim] = count
        axis = x.perm.index(dim)
        begin = [0] * x.rank
        steps = [1] * x.rank
        begin[axis], steps[axis] = start, step
        ir_shape = tuple(shape[p] for p in x.perm)
        y = self.add("slice", [x.name], ir_shape, {"begin": tuple(begin), "step": tuple(steps)})
        return _Value(y, x.perm, tuple(shape))

    def op_slice(self, node, args, kwargs, meta):
        x = args[0]
        dim = (args[1] if len(args) > 1 else 0) % x.rank
        start = args[2] if len(args) > 2 else None
        stop = args[3] if len(args) > 3 else None
        step = args[4] if len(args) > 4 else 1
        if stop is not None and stop > 2 ** 62:
            stop = None
        return self.slice_value(x, dim, start, stop, step)

    def op_narrow(self, node, args, kwargs, meta):
        x, dim, start, length = args[0], args[1] % args[0].rank, args[2], args[3]
        return self.slice_value(x, dim, start, start + length)

    def op_select(self, node, args, kwargs, meta):
        x, dim, idx = args[0], args[1] % args[0].rank, args[2]
        idx = idx % x.shape[dim]
        return self.drop_dims(self.slice_value(x, dim, idx, idx + 1), [dim])

    def op_split(self, node, args, kwargs, meta):
        x = args[0]
        sizes = args[1]
        dim = (args[2] if len(args) > 2 else kwargs.get("dim", 0)) % x.rank
        n = x.shape[dim]
        if isinstance(sizes, int):
            sizes = [sizes] * ((n + sizes - 1) // sizes)
            sizes[-1] = n - sum(sizes[:-1])
        out, start = [], 0
        for s in sizes:
            out.append(self.slice_value(x, dim, start, start + s))
            start += s
        return tuple(out)

    op_split_with_sizes = op_split

    def op_chunk(self, node, args, kwargs, meta):
        x, chunks = args[0], args[1]
        dim = (args[2] if len(args) > 2 else kwargs.get("dim", 0)) % x.rank
        size = (x.shape[dim] + chunks - 1) // chunks
        return self.op_split(node, [x, size, dim], {}, meta)

    def op_unbind(self, node, args, kwargs, meta):
        x = args[0]
        dim = (args[1] if len(args) > 1 else kwargs.get("dim", 0)) % x.rank
        out = []
        for i in range(x.shape[dim]):
            out.append(self.drop_dims(self.slice_value(x, dim, i, i + 1), [dim]))
        return tuple(out)

    def op_sym_size(self, node, args, kwargs, meta):
        return args[0].shape[args[1]]

    def op_pad(self, node, args, kwargs, meta):
        mode = args[2] if len(args) > 2 else kwargs.get("mode", "constant")
        if mode != "constant":
            self.fail("%s padding" % mode)
        value = args[3] if len(args) > 3 else kwargs.get("value", 0.0)
        return self.constant_pad(args[0], args[1], 0.0 if value is None else value)

    def op_constant_pad_nd(self, node, args, kwargs, meta):
        return self.constant_pad(args[0], args[1], args[2] if len(args) > 2 else
                                 kwargs.get("value", 0.0))

    def constant_pad(self, x, pads, value):
        pads = list(pads) + [0] * (8 - len(pads))
        l, r, t, b = pads[:4]
        if any(pads[4:]):
            self.fail("padding of the channel or batch dimension")
        if min(l, r, t, b) < 0:
            self.fail("negative padding (cropping)")
        x4, back = self.to_nhwc(x) if x.rank in (3, 4) else self.fail("pad of a rank-%d tensor" % x.rank)
        if x.rank == 3:
            t, b = 0, 0
        y = self.pad_node(x4, (t, b, l, r), value)
        return back(y.name, y.shape[1], y.shape[2], y.shape[3])

    def op_mean(self, node, args, kwargs, meta):
        x = args[0]
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return self.reduce_mean(x, dims, keep, meta)

    def reduce_mean(self, x, dims, keep, meta=None):
        if x.flat is not None:
            x = self.ident(x)
        if dims is None or dims == []:
            dims = list(range(x.rank))
        dims = sorted(set(d % x.rank for d in (dims if isinstance(dims, (list, tuple)) else [dims])))
        kept_torch = [d for d in range(x.rank) if d not in dims]
        out_shape = tuple(1 if d in dims else x.shape[d] for d in range(x.rank)) if keep else \
            tuple(x.shape[d] for d in kept_torch)
        if x.rank == 4 and x.perm == NHWC and dims == [2, 3]:
            ir = (1, 1, 1, x.shape[1]) if keep else (1, x.shape[1])
            y = self.add("global_avg_pool", [x.name], ir)
            return _Value(y, NHWC if keep else (0, 1), out_shape)
        axes = tuple(x.perm.index(d) for d in dims if d in x.perm)
        if keep:
            perm = x.perm
        else:
            perm = tuple(kept_torch.index(p) for p in x.perm if p not in dims)
        if not axes:
            return _Value(x.name, perm, out_shape)
        ir = tuple(1 if i in axes else d for i, d in enumerate(x.ir_shape)) if keep else \
            (tuple(d for i, d in enumerate(x.ir_shape) if i not in axes) or (1,))
        y = self.add("mean", [x.name], ir, {"axes": axes})
        return _Value(y, perm, out_shape)

    def op_sum(self, node, args, kwargs, meta):
        x = args[0]
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if dims is None or dims == []:
            dims = list(range(x.rank))
        dl = dims if isinstance(dims, (list, tuple)) else [dims]
        count = int(np.prod([x.shape[d % x.rank] for d in dl]))
        return self.binary("mul", self.reduce_mean(x, dims, keep, meta), np.float32(count))

    def op_softmax(self, node, args, kwargs, meta, log=False):
        x = args[0]
        dim = (args[1] if len(args) > 1 else kwargs.get("dim", -1)) % x.rank
        if x.flat is not None:
            x = self.ident(x)
        if x.perm[-1] != dim:
            perm = tuple(p for p in x.perm if p != dim) + (dim,)
            x = self.layout(x, perm)
        y = self.add("softmax", [x.name], x.ir_shape, {"beta": 1.0})
        v = _Value(y, x.perm, x.shape)
        return self.unary(v, "log") if log else v

    op__softmax = op_softmax

    def op_log_softmax(self, node, args, kwargs, meta):
        return self.op_softmax(node, args, kwargs, meta, log=True)

    op__log_softmax = op_log_softmax

    # -- recurrent ---------------------------------------------------------------------

    def rnn(self, kind, args, meta):
        x, hx, params, has_biases, layers, _, train, bidir, batch_first = args[:9]
        if bidir:
            self.fail("bidirectional %s" % kind)
        if train:
            self.fail("%s in training mode" % kind)
        hx = hx if isinstance(hx, (list, tuple)) else [hx]
        for h in hx:
            if isinstance(h, _Value) or np.any(np.asarray(h) != 0):
                self.fail("%s with a non-zero initial state" % kind)
        if x.flat is not None:
            x = self.ident(x)
        if x.rank != 3:
            self.fail("%s on an unbatched input" % kind)
        seq = self.reshape(x, (1, x.shape[1], x.shape[2]) if batch_first else
                           (1, x.shape[0], x.shape[2]))
        if (batch_first and x.shape[0] != 1) or (not batch_first and x.shape[1] != 1):
            self.fail("%s with batch > 1" % kind)
        per = 4 if has_biases else 2
        T = seq.shape[1]
        lasts = []
        for layer in range(layers):
            p = [np.asarray(t, np.float32) for t in params[layer * per:(layer + 1) * per]]
            w_ih, w_hh = p[0], p[1]
            G = 4 if kind == "lstm" else 3
            H = w_hh.shape[1]
            b_ih = p[2] if has_biases else np.zeros(G * H, np.float32)
            b_hh = p[3] if has_biases else np.zeros(G * H, np.float32)
            if kind == "lstm":
                ins = [seq.name, self.const(w_ih.T, "W%d" % layer), self.const(w_hh.T, "U%d" % layer),
                       self.const(b_ih + b_hh, "b%d" % layer)]
            else:  # PyTorch gate order r, z, n -> pTensor z, r, n
                order = np.concatenate([np.arange(H, 2 * H), np.arange(0, H), np.arange(2 * H, 3 * H)])
                ins = [seq.name, self.const(w_ih[order].T, "W%d" % layer),
                       self.const(w_hh[order].T, "U%d" % layer),
                       self.const(b_ih[order], "bx%d" % layer), self.const(b_hh[order], "bh%d" % layer)]
            y = self.add(kind, ins, (1, T, H), base="%s_l%d" % (self.node_name, layer))
            seq = _Value(y, (0, 1, 2), (1, T, H))
            last = self.slice_value(seq, 1, T - 1, T)
            lasts.append(self.reshape(last, (1, 1, H)))
        out = seq if batch_first else self.reshape(seq, (T, 1, seq.shape[2]))
        h_n = lasts[0] if layers == 1 else self.op_cat(None, [lasts, 0], {}, None)
        if kind == "lstm":
            return (out, h_n, _Unavailable("LSTM cell state c_n"))
        return (out, h_n)

    def op_lstm(self, node, args, kwargs, meta):
        if not isinstance(args[1], (list, tuple)):
            self.fail("packed-sequence LSTM")
        return self.rnn("lstm", args, meta)

    def op_gru(self, node, args, kwargs, meta):
        return self.rnn("gru", args, meta)
