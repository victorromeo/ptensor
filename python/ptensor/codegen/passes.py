"""Graph transformations, each optional and selected by a CodeGenerator flag.

    prune(graph, ratio)              structured (filter / unit) pruning
    fuse(graph)                      fold batch norm and activations into the layer before
    encode_weights(graph, mode)      store weights packed / Hough / polynomial (float activations)
    quantize_int8(graph, calib)      full-integer post-training quantisation
    plan_memory(graph)               static arena offsets for every activation

Every pass returns a new graph (the input is not modified) and appends a line
to ``graph.notes`` saying what it did.
"""
import numpy as np

from .ir import (FLOAT32, INT8, INT32, LINEAR_OPS, Node, QuantParams, canonical_activation,
                 clamp_range)

_CLAMP_KINDS = ("relu", "relu6", "relu1", "clip")


def _kind_range(attrs):
    k = attrs["kind"]
    if k == "clip":
        return float(attrs["lo"]), float(attrs["hi"])
    return clamp_range(k)


def _compose(first, second):
    lo1, hi1 = clamp_range(first)
    lo2, hi2 = second
    lo, hi = max(lo1, lo2), min(hi1, hi2)
    if lo > hi:
        lo = hi = lo2 if hi1 < lo2 else hi2
    return canonical_activation((lo, hi))


def _single_reader(g, name):
    return len(g.consumers(name)) == 1 and name not in g.outputs


def _same_quant(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (np.array_equal(a.scale, b.scale) and np.array_equal(a.zero_point, b.zero_point))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fold_batch_norm_params(mean, var, gamma, beta, eps):
    scale = (gamma / np.sqrt(var.astype(np.float64) + eps)).astype(np.float32)
    shift = (beta - mean * scale).astype(np.float32)
    return scale, shift


def fuse(graph):
    """Fold every batch_norm that directly follows a conv2d / depthwise_conv2d /
    dense into its weights and bias, and absorb clamp activations (ReLU,
    ReLU6, ReLU1, clip) into the conv / depthwise / dense / add before them.

    The emitter then uses pTensor's fused operators (one pass per layer, no
    intermediate tensor). Float graphs are folded exactly; int8 graphs only
    absorb an activation whose quantisation matches the layer output's.
    """
    g = graph.copy()
    folded, absorbed = [], []
    changed = True
    while changed:
        changed = False
        for node in list(g.nodes):
            x = node.inputs[0] if node.inputs else None
            prod = g.producer(x) if x is not None else None
            if prod is None or not _single_reader(g, x):
                continue
            if node.op == "batch_norm" and prod.op in LINEAR_OPS \
                    and prod.attrs.get("activation") is None and not g.is_quantized():
                mean, var, gamma, beta = (g.tensors[n].data for n in node.inputs[1:5])
                scale, shift = fold_batch_norm_params(mean, var, gamma, beta,
                                                      node.attrs.get("epsilon", 1e-3))
                w = g.tensors[prod.inputs[1]]
                b = g.tensors[prod.inputs[2]]
                if w.encoding is not None:
                    raise ValueError("fuse() must run before encode_weights()")
                if prod.op == "conv2d":
                    wf = w.data * scale[:, None, None, None]
                else:  # dense [in, out], depthwise [1, kh, kw, C]: last axis
                    wf = w.data * scale
                nw = g.add_const(w.name + "_bn", wf.astype(np.float32))
                nb = g.add_const(b.name + "_bn", (b.data * scale + shift).astype(np.float32))
                prod.inputs[1], prod.inputs[2] = nw.name, nb.name
                prod.outputs[0] = node.outputs[0]
                g.nodes.remove(node)
                g.bn_folded = getattr(g, "bn_folded", set()) | {prod.name}
                folded.append("%s<-%s" % (prod.name, node.name))
                changed = True
                break
            if node.op == "activation" and node.attrs["kind"] in _CLAMP_KINDS \
                    and prod.op in LINEAR_OPS + ("add",):
                if g.is_quantized():
                    if not _same_quant(g.tensors[x].quant, g.tensors[node.outputs[0]].quant):
                        continue
                    if node.attrs["kind"] == "clip":
                        continue
                prod.attrs["activation"] = _compose(prod.attrs.get("activation"),
                                                    _kind_range(node.attrs))
                prod.outputs[0] = node.outputs[0]
                g.nodes.remove(node)
                absorbed.append("%s<-%s" % (prod.name, node.name))
                changed = True
                break
    g.remove_unused()
    g.notes.append("fuse: folded %d batch norm(s), absorbed %d activation(s)"
                   % (len(folded), len(absorbed)))
    g.fused = True
    return g


# ---------------------------------------------------------------------------
# Structured pruning
# ---------------------------------------------------------------------------

_PASS_THROUGH = ("batch_norm", "activation", "max_pool2d", "avg_pool2d", "global_avg_pool")


def _prune_chain(g, prod):
    """Follow a producer's output channels to the layer that reads them.

    Returns (chain, sink) where chain lists the nodes the channels pass
    through, or None when the channels cannot be removed safely (the output is
    a graph output, is read twice, or reaches an op that mixes channels in a
    way we do not rewrite, such as add)."""
    t = prod.outputs[0]
    chain = []
    while True:
        if not _single_reader(g, t):
            return None
        nxt = g.consumers(t)[0]
        if nxt.inputs[0] != t:
            return None
        if nxt.op in ("conv2d", "dense"):
            return chain, nxt
        if nxt.op in _PASS_THROUGH or (nxt.op == "depthwise_conv2d"
                                       and nxt.attrs.get("depth_multiplier", 1) == 1):
            chain.append(nxt)
        elif nxt.op == "reshape":
            src, dst = g.tensors[t].shape, g.tensors[nxt.outputs[0]].shape
            if len(dst) != 2 or dst[1] != int(np.prod(src[1:])):
                return None
            chain.append(nxt)
        else:
            return None
        t = nxt.outputs[0]


def _scores(w, op, criterion):
    axis_data = np.moveaxis(w, 0 if op == "conv2d" else -1, 0).reshape(
        w.shape[0] if op == "conv2d" else w.shape[-1], -1)
    if criterion == "l1":
        return np.abs(axis_data).sum(1)
    if criterion == "l2":
        return np.sqrt((axis_data.astype(np.float64) ** 2).sum(1))
    raise ValueError("criterion must be 'l1' or 'l2'")


def _n_keep(n, ratio, multiple_of, min_keep):
    keep = int(round(n * (1.0 - ratio)))
    if multiple_of > 1:
        keep = int(np.ceil(keep / float(multiple_of)) * multiple_of)
    return int(min(n, max(min_keep, keep)))


def prune(graph, ratio, criterion="l1", multiple_of=1, min_keep=1):
    """Remove the lowest-magnitude output channels (conv filters, dense units)
    of every layer whose channels feed exactly one later conv / dense, and the
    matching input channels of that layer. The result is a smaller dense
    network, so no kernel changes are needed.

    ratio: fraction to remove, one float for all layers or a dict
    {layer name: ratio}. Without fine-tuning accuracy drops; for the best
    result prune and fine-tune in Keras with ptensor.prune.PrunedModel and
    give the generator the pruned model (ratio 0).
    """
    g = graph.copy()
    if g.is_quantized():
        raise ValueError("prune the float model before int8 quantisation")
    done = []
    for prod in list(g.nodes):
        if prod.op not in ("conv2d", "dense"):
            continue
        r = ratio.get(prod.name, 0.0) if isinstance(ratio, dict) else float(ratio)
        if r <= 0:
            continue
        found = _prune_chain(g, prod)
        if found is None:
            continue
        chain, sink = found
        w = g.tensors[prod.inputs[1]]
        if w.encoding is not None:
            raise ValueError("prune() must run before encode_weights()")
        n = w.shape[0] if prod.op == "conv2d" else w.shape[1]
        keep = np.sort(np.argsort(-_scores(w.data, prod.op, criterion), kind="stable")[
            :_n_keep(n, r, multiple_of, min_keep)])
        k = len(keep)
        if k == n:
            continue

        def sliced_const(name, arr):
            return g.add_const(name + "_pruned", np.ascontiguousarray(arr)).name

        if prod.op == "conv2d":
            prod.inputs[1] = sliced_const(w.name, w.data[keep])
        else:
            prod.inputs[1] = sliced_const(w.name, w.data[:, keep])
        b = g.tensors[prod.inputs[2]]
        prod.inputs[2] = sliced_const(b.name, b.data[keep])

        def shrink(tname):
            t = g.tensors[tname]
            t.shape = t.shape[:-1] + (k,)

        shrink(prod.outputs[0])
        spatial = None
        for node in chain:
            if node.op == "batch_norm":
                for i in range(1, 5):
                    c = g.tensors[node.inputs[i]]
                    node.inputs[i] = sliced_const(c.name, c.data[keep])
            elif node.op == "depthwise_conv2d":
                f = g.tensors[node.inputs[1]]
                node.inputs[1] = sliced_const(f.name, f.data[..., keep])
                bb = g.tensors[node.inputs[2]]
                node.inputs[2] = sliced_const(bb.name, bb.data[keep])
            if node.op == "reshape":
                src = g.tensors[node.inputs[0]].shape
                spatial = int(np.prod(src[1:-1]))
                g.tensors[node.outputs[0]].shape = (1, spatial * k)
            else:
                shrink(node.outputs[0])
        sw = g.tensors[sink.inputs[1]]
        if sink.op == "conv2d":
            sink.inputs[1] = sliced_const(sw.name, sw.data[..., keep])
        elif spatial is not None:
            rows = (np.arange(spatial)[:, None] * n + keep[None, :]).reshape(-1)
            sink.inputs[1] = sliced_const(sw.name, sw.data[rows])
        else:
            sink.inputs[1] = sliced_const(sw.name, sw.data[keep])
        done.append("%s %d->%d" % (prod.name, n, k))
    g.remove_unused()
    g.notes.append("prune: %s" % (", ".join(done) if done else "nothing prunable"))
    return g


# ---------------------------------------------------------------------------
# Weight storage
# ---------------------------------------------------------------------------

def encode_weights(graph, mode, overrides=None, layers=None, min_size=64):
    """Store the weights of every conv2d / depthwise_conv2d / dense in a pTensor
    weight format (``ptensor.qat`` mode name or WeightFormat: "float16",
    "uint4", "int8", "hough4", "cubic4", ...). Activations stay float32; the
    device decodes each weight as it reads it.

    overrides: {layer name: Encoded} for weights already encoded elsewhere,
    e.g. ``{layer.name: enc for layer, enc, _ in qat_model.export()}`` so the
    generated code uses exactly the quantisation QAT trained with.
    layers: only encode these layer names. Weights smaller than min_size
    elements stay float32 (the codebook would cost more than it saves).
    """
    from .. import qat
    g = graph.copy()
    if g.is_quantized():
        raise ValueError("encode_weights is for float graphs; int8 graphs already store int8 weights")
    fmt = qat.get_format(mode) if mode is not None else None
    overrides = dict(overrides or {})
    done = {}
    for node in g.nodes:
        if node.op not in LINEAR_OPS:
            continue
        w = g.tensors[node.inputs[1]]
        if w.name in done:
            continue
        enc = None
        if node.name in overrides:
            if node.name in getattr(g, "bn_folded", ()):
                raise ValueError("%s had a batch norm folded into its weights, so its pre-encoded "
                                 "weights no longer apply: fold batch norms before QAT, or "
                                 "generate with fuse=False" % node.name)
            enc = overrides.pop(node.name)
            if tuple(enc.shape) != w.shape:
                raise ValueError("override for %s has shape %s, weights are %s"
                                 % (node.name, enc.shape, w.shape))
        elif fmt is not None and (layers is None or node.name in layers) and w.size >= min_size:
            enc = fmt.quantizer(w.shape).encode(w.data)
        if enc is not None and enc.kind == "f32":
            enc = None
        if enc is not None:
            nw = g.add_const(w.name + "_enc", np.asarray(enc.decode(), np.float32).reshape(w.shape))
            nw.encoding = enc
            node.inputs[1] = nw.name
            done[w.name] = enc.describe()
    if overrides:
        raise ValueError("overrides name layers that are not in the graph: %s" % sorted(overrides))
    g.remove_unused()
    g.notes.append("encode_weights(%s): %d tensor(s) encoded" % (getattr(fmt, "name", mode), len(done)))
    return g


# ---------------------------------------------------------------------------
# Full-integer int8 quantisation
# ---------------------------------------------------------------------------

def _act_qparams(lo, hi):
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = np.float32((hi - lo) / 255.0 or 1e-8)
    zp = int(np.clip(np.rint(-128 - lo / scale), -128, 127))
    return QuantParams(scale, zp)


def _weight_q(w, axis):
    moved = np.moveaxis(w, axis, 0)
    per = np.abs(moved.reshape(moved.shape[0], -1)).max(1)
    scales = (per / 127.0).astype(np.float32)
    scales[scales == 0] = 1e-8
    shape = [1] * w.ndim
    shape[axis] = -1
    q = np.clip(np.rint(w / scales.reshape(shape)), -127, 127).astype(np.int8)
    return q, scales


def quantize_int8(graph, calibration, input_scale=None, input_zero_point=None,
                  float_io=False):
    """Post-training full-integer quantisation, the way the TFLite converter
    does it: int8 activations (asymmetric, range from calibration), int8
    weights (symmetric, per output channel) and int32 biases.

    calibration: iterable of float inputs (one example each, any shape that
    reshapes to the graph input). input_scale / input_zero_point fix the
    input quantisation (e.g. 1/255, -128 for 0..1 images); otherwise it comes
    from calibration. float_io=True adds quantize / dequantize at the graph
    boundary so the generated function takes and returns floats.

    The graph must be fused (activations folded into the layer before),
    which CodeGenerator does automatically for int8.
    """
    from . import reference
    if graph.is_quantized():
        return graph
    if len(graph.inputs) != 1:
        raise NotImplementedError("int8 quantisation of multi-input graphs")
    for n in graph.nodes:
        if n.op in ("batch_norm", "avg_pool2d", "global_avg_pool") or \
                (n.op == "activation" and n.attrs["kind"] in ("sigmoid", "tanh", "clip")):
            raise NotImplementedError("int8 %s (%s) has no TFLite-exact pTensor kernel; keep "
                                      "the model float or run fuse() first" % (n.op, n.name))
        if n.op in LINEAR_OPS + ("add",) and n.attrs.get("activation") not in (None, "relu", "relu6", "relu1"):
            raise NotImplementedError("int8 %s with clamp %s" % (n.name, n.attrs.get("activation")))
    lo, hi = {}, {}
    count = 0
    for x in calibration:
        _, env = reference.run(graph, x, collect=True)
        for k, v in env.items():
            lo[k] = min(lo.get(k, np.inf), float(v.min()))
            hi[k] = max(hi.get(k, -np.inf), float(v.max()))
        count += 1
    if not count:
        raise ValueError("calibration data is empty")

    g = graph.copy()
    q = {}
    inp = g.inputs[0]
    if input_scale is not None:
        q[inp] = QuantParams(input_scale, 0 if input_zero_point is None else input_zero_point)
    else:
        q[inp] = _act_qparams(lo[inp], hi[inp])
    for n in g.nodes:
        out = n.outputs[0]
        if n.op in ("max_pool2d", "reshape"):
            q[out] = q[n.inputs[0]]
        elif n.op == "softmax":
            q[out] = QuantParams(1.0 / 256.0, -128)
        else:
            q[out] = _act_qparams(lo[out], hi[out])
    for n in g.nodes:
        if n.op not in LINEAR_OPS:
            continue
        w = g.tensors[n.inputs[1]]
        axis = {"conv2d": 0, "depthwise_conv2d": 3, "dense": 1}[n.op]
        wq, ws = _weight_q(w.data, axis)
        in_scale = q[n.inputs[0]].scale[0]
        b = g.tensors[n.inputs[2]]
        bq = np.rint(b.data.astype(np.float64) / (in_scale * ws.astype(np.float64))).astype(np.int32)
        n.inputs[1] = g.add_const(w.name + "_q", wq, INT8, QuantParams(ws, 0, axis)).name
        n.inputs[2] = g.add_const(b.name + "_q", bq, INT32,
                                  QuantParams((in_scale * ws).astype(np.float32), 0, 0)).name
    for name, qp in q.items():
        t = g.tensors[name]
        t.dtype, t.quant = INT8, qp
    if float_io:
        fin = g.add_tensor(inp + "_float", g.tensors[inp].shape, FLOAT32)
        g.nodes.insert(0, Node("quantize", [fin.name], [inp], name="quantize_input"))
        g.inputs = [fin.name]
        outs = []
        for o in g.outputs:
            fo = g.add_tensor(o + "_float", g.tensors[o].shape, FLOAT32)
            g.add_node("dequantize", [o], [fo], name="dequantize_" + o)
            outs.append(fo.name)
        g.outputs = outs
    g.remove_unused()
    g.notes.append("quantize_int8: %d calibration example(s), input scale %g zero point %d"
                   % (count, q[inp].scale[0], q[inp].zero_point[0]))
    return g


# ---------------------------------------------------------------------------
# Static memory planning
# ---------------------------------------------------------------------------

class MemoryPlan(object):
    """Arena offsets for every activation tensor.

    offsets[name]   byte offset in the arena
    arena_bytes     size of the arena
    naive_bytes     one buffer per activation
    live_peak       most bytes live at once (a lower bound without in-place sharing)
    shared          tensors that reuse their input's bytes (reshape views, in-place ops)
    """

    def __init__(self):
        self.offsets, self.shared = {}, {}
        self.arena_bytes = self.naive_bytes = self.live_peak = 0
        self.alignment = 8

    def describe(self):
        return ("planned arena %d B (naive %d B, live peak %d B, %d shared buffer(s))"
                % (self.arena_bytes, self.naive_bytes, self.live_peak, len(self.shared)))


def _inplace_ok(g, node):
    """Can this node write its output over its first input?"""
    x, y = g.tensors[node.inputs[0]], g.tensors[node.outputs[0]]
    if x.dtype != y.dtype or y.nbytes > x.nbytes:
        return False
    if node.op in ("activation", "batch_norm"):
        return True
    if node.op == "add":
        return x.shape == y.shape
    if node.op == "softmax":  # the int8 kernel re-reads its input after writing
        return x.dtype == FLOAT32
    return False


def plan_memory(graph, alignment=8, inplace=True):
    """Greedy-by-size placement (what TFLite Micro and pTensor's ArenaPlanner
    use) of every activation, with reshape outputs as views of their input and
    elementwise outputs written in place when the input is dead afterwards."""
    g = graph
    acts = g.activations()
    n_nodes = len(g.nodes)
    first, last = {}, {}
    for name in acts:
        first[name] = -1 if name in g.inputs else None
        last[name] = n_nodes if name in g.outputs else -1
    for i, node in enumerate(g.nodes):
        for o in node.outputs:
            if first.get(o) is None:
                first[o] = i
        for t in node.inputs:
            if t in last:
                last[t] = max(last[t], i)
    for name in acts:
        if first[name] is None:
            first[name] = 0
        if last[name] < first[name]:
            last[name] = first[name]

    root = {name: name for name in acts}

    def find(a):
        while root[a] != a:
            a = root[a]
        return a

    members = {name: [name] for name in acts}
    glast = dict(last)
    plan = MemoryPlan()
    plan.alignment = alignment
    for i, node in enumerate(g.nodes):
        if not node.inputs or node.inputs[0] not in root:
            continue
        r = find(node.inputs[0])
        out = node.outputs[0]
        if node.op == "reshape":
            ok = True
        elif inplace and _inplace_ok(g, node):
            ok = (glast[r] == i and not any(m in g.inputs for m in members[r])
                  and not any(m in g.outputs for m in members[r]))
        else:
            ok = False
        if ok:
            root[out] = r
            members[r].extend(members.pop(out))
            glast[r] = max(glast[r], glast.pop(out))
            plan.shared[out] = node.inputs[0]

    def aligned(n):
        return (n + alignment - 1) // alignment * alignment

    groups = []
    for r, ms in members.items():
        size = aligned(max(g.tensors[m].nbytes for m in ms))
        start = min(first[m] for m in ms)
        groups.append((size, start, glast[r], r))
    groups.sort(key=lambda t: (-t[0], t[1]))
    placed = []
    for size, start, end, r in groups:
        offset = 0
        for (o2, s2, a2, b2) in sorted(placed):
            if a2 > end or b2 < start:
                continue
            if offset + size <= o2:
                break
            offset = max(offset, o2 + s2)
        placed.append((offset, size, start, end))
        for m in members[r]:
            plan.offsets[m] = offset
        plan.arena_bytes = max(plan.arena_bytes, offset + size)
    plan.naive_bytes = sum(aligned(g.tensors[m].nbytes) for m in acts)
    for i in range(-1, n_nodes + 1):
        plan.live_peak = max(plan.live_peak, sum(s for (_, s, a, b) in placed if a <= i <= b))
    return plan


def unfuse(graph):
    """Split fused clamp activations off conv / depthwise / dense / add into
    activation nodes of their own (float graphs). Used when fusion is off so
    every operator runs as the plain reference kernel."""
    g = graph.copy()
    if g.is_quantized():
        return g  # int8 kernels always apply their activation themselves
    split = 0
    for node in list(g.nodes):
        act = node.attrs.get("activation")
        if node.op not in LINEAR_OPS + ("add",) or act is None:
            continue
        y = g.tensors[node.outputs[0]]
        mid = g.add_tensor(y.name + "_preact", y.shape, y.dtype)
        node.outputs[0] = mid.name
        node.attrs["activation"] = None
        lo, hi = clamp_range(act)
        name = canonical_activation(act)
        attrs = {"kind": name} if isinstance(name, str) else {"kind": "clip", "lo": lo, "hi": hi}
        g.nodes.insert(g.nodes.index(node) + 1,
                       Node("activation", [mid.name], [y.name], attrs, node.name + "_act"))
        split += 1
    if split:
        g.notes.append("unfuse: %d activation(s) run as separate operators" % split)
    return g
