"""C/C++ emitter: an ``ir.Graph`` becomes a pTensor model.

Files written for a model called ``mnist``:

    mnist.h      C header: mnist_init(), mnist_invoke(), sizes, int8 I/O parameters
    mnist.cpp    the model: weights as const arrays (flash), the activation
                 arena, and one pTensor operator call per graph node
    mnist.cmake  the pTensor features this model needs (PTENSOR_WITH_*) and a
                 static library target; include it before add_subdirectory(ptensor)

The .cpp includes only the operator and tensor headers the graph uses, and
the .cmake turns off every optional library feature it does not use, so the
firmware carries only what this model needs (pTensor goal 7).
"""
import re

import numpy as np

from .ir import FLOAT32, INT8, INT32, LINEAR_OPS, clamp_range

_TTYPE = {FLOAT32: "flt", INT8: "i8", INT32: "i32"}
_CTYPE = {FLOAT32: "float", INT8: "int8_t", INT32: "int32_t"}
_TFLM_ACT = {None: 0, "relu": 1, "relu1": 2, "relu6": 3}
FEATURES = ("PTENSOR_WITH_QUANTIZED_OPS", "PTENSOR_WITH_FUSION", "PTENSOR_WITH_HOUGH",
            "PTENSOR_WITH_POLY_QUANT", "PTENSOR_WITH_FAST_KERNELS",
            "PTENSOR_WITH_SIMPLE_ERROR_HANDLER")


def c_ident(s):
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    if not s or s[0].isdigit():
        s = "_" + s
    return s


def _fmt_float(v):
    v = float(np.float32(v))
    if not np.isfinite(v):
        raise ValueError("non-finite constant %r cannot be emitted" % v)
    s = "%.9g" % v
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"


def _array_body(values, kind, per_line):
    if kind == "float":
        items = [_fmt_float(v) for v in values]
    else:
        items = [str(int(v)) for v in values]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i:i + per_line]) + ",")
    return "\n".join(lines) if lines else "    0,"


class EmitterError(NotImplementedError):
    pass


class CEmitter(object):
    """Writes the files for one graph. Use through CodeGenerator."""

    def __init__(self, graph, name, plan=None, fused=False, weights_in_ram=False,
                 fast_kernels=True, meta_arena_bytes=None, ram_arena_bytes=None,
                 options_comment=""):
        self.g = graph
        self.name = c_ident(name)
        self.NAME = self.name.upper()
        self.plan = plan            # MemoryPlan, or None for dynamic RamTensors
        self.fused = fused
        self.weights_in_ram = weights_in_ram
        self.fast_kernels = fast_kernels
        self.meta_arena_bytes = meta_arena_bytes
        self.ram_arena_bytes = ram_arena_bytes
        self.options_comment = options_comment
        self.includes = set(["pTensor/core/context.hpp", "pTensor/core/errorHandler.hpp",
                             "pTensor/allocators/arenaAllocator.hpp",
                             "pTensor/tensors/BufferTensor.hpp"])
        self.features = set()
        self.const_ids = {}
        self.const_names = {}
        self.decls = []
        self.init_lines = []
        self.flash_bytes = 0
        self.ram_weight_bytes = 0
        self.max_scratch = 0

    # -- constants ------------------------------------------------------------

    def _array(self, ident, ctype, values, kind, per_line, comment=None):
        n = len(values)
        attr = "%s_WEIGHT_ATTR" % self.NAME
        head = "// %s\n" % comment if comment else ""
        self.decls.append("%salignas(8) const %s %s[%d] %s = {\n%s\n};"
                          % (head, ctype, ident, max(n, 1), attr,
                             _array_body(values, kind, per_line)))
        self.flash_bytes += n * {"float": 4, "int8_t": 1, "uint8_t": 1, "int32_t": 4}[ctype]

    def const(self, name):
        """Emit tensor `name` once; return a C++ expression that makes a
        TensorInterface* for it."""
        if name in self.const_ids:
            return self.const_ids[name]
        t = self.g.tensors[name]
        ident = ("k%d_%s" % (len(self.const_names), c_ident(name)))[:60]
        self.const_names[name] = ident
        shape = self.shape(t.shape)
        enc = t.encoding
        if enc is None:
            ctype = _CTYPE[t.dtype]
            kind = "float" if t.dtype == FLOAT32 else "int"
            self._array(ident, ctype, t.data.reshape(-1), kind, 8 if kind == "float" else 16,
                        "%s %s %s" % (name, t.shape, t.dtype))
            self.includes.add("pTensor/tensors/RomTensor.hpp")
            expr = "new RomTensor(%s, %s, %s)" % (shape, _TTYPE[t.dtype], ident)
        else:
            expr = self._encoded(name, t, ident, shape)
        self.const_ids[name] = expr
        return expr

    def _encoded(self, name, t, ident, shape):
        from .. import qat
        enc = t.encoding
        comment = "%s %s stored as %s (%d B)" % (name, t.shape, enc.describe(), enc.nbytes)
        if isinstance(enc, qat.Packed):
            payload = np.asarray(enc.payload, np.uint8).reshape(-1)
            self._array(ident, "uint8_t", payload, "int", 16, comment)
            self.decls.append("const PackedEncoding %s_enc = {static_cast<PackedFormat>(%d), %d, %s, %d};"
                              % (ident, enc.fmt, enc.bits, _fmt_float(enc.scale), enc.zero_point))
            self.includes.add("pTensor/tensors/PackedRomTensor.hpp")
            self.flash_bytes += 12
            make = "new PackedRomTensor(%s, flt, %s_enc, %s)" % (shape, ident, ident)
        elif isinstance(enc, qat.HoughEncoded):
            c = enc.code
            self._array(ident, "uint8_t", np.asarray(c.stream(), np.uint8).reshape(-1), "int", 16, comment)
            self._array(ident + "_lines", "float", np.asarray(c.lines, np.float32).reshape(-1),
                        "float", 8)
            self.decls.append("const HoughEncoding %s_enc = {%d, %d, %d, %s_lines};"
                              % (ident, c.line_bits, c.t_bits, c.stride, ident))
            self.includes.add("pTensor/tensors/HoughRomTensor.hpp")
            self.features.add("PTENSOR_WITH_HOUGH")
            make = "new HoughRomTensor(%s, %s_enc, %s)" % (shape, ident, ident)
        elif isinstance(enc, qat.PolyEncoded):
            c = enc.code
            self._array(ident, "uint8_t", np.asarray(c.stream(), np.uint8).reshape(-1), "int", 16, comment)
            lut = "nullptr"
            if c.use_lut:
                self._array(ident + "_lut", "float", np.asarray(c.lut(), np.float32).reshape(-1),
                            "float", 8)
                lut = ident + "_lut"
            coeffs = ", ".join(_fmt_float(v) for v in np.asarray(c.coeffs, np.float32).reshape(-1)[:4])
            self.decls.append("const PolyEncoding %s_enc = {%d, %d, {%s}, %s};"
                              % (ident, c.degree, c.bits, coeffs, lut))
            self.includes.add("pTensor/tensors/PolyRomTensor.hpp")
            self.features.add("PTENSOR_WITH_POLY_QUANT")
            make = "new PolyRomTensor(%s, %s_enc, %s)" % (shape, ident, ident)
        else:
            raise EmitterError("no storage for encoding %s" % type(enc).__name__)
        if not self.weights_in_ram:
            return make
        # Decode once at init into RAM, then run the dense float fast paths.
        self.includes.add("pTensor/tensors/RomTensor.hpp")
        self.decls.append("alignas(8) float %s_ram[%d];" % (ident, t.size))
        self.ram_weight_bytes += 4 * t.size
        self.init_lines.append(
            "  {\n    Tensor t = %s;\n    const Tensor& ct = t;\n"
            "    for (uint32_t i = 0; i < %du; ++i) %s_ram[i] = static_cast<float>(ct(i));\n  }"
            % (make, t.size, ident))
        return "new RomTensor(%s, flt, %s_ram)" % (shape, ident)

    def weight_quant(self, name, out_axis_len):
        """Statement setting per-channel int8 weight parameters on tensor `w`."""
        t = self.g.tensors[name]
        q = t.quant
        scales = np.asarray(q.scale, np.float32)
        if scales.size == 1:
            scales = np.repeat(scales, out_axis_len)
        ident = self.const_names[name]
        key = ident + "_qs"
        if key not in self.const_ids:
            self._array(ident + "_scales", "float", scales, "float", 8)
            self._array(ident + "_zps", "int32_t", np.zeros(scales.size, np.int32), "int", 16)
            self.const_ids[key] = True
        return ("w->set_quantization_params(PerChannelQuantizationParams(%s_zps, %s_scales, %d));"
                % (ident, ident, scales.size))

    # -- activations -----------------------------------------------------------

    @staticmethod
    def shape(dims):
        for d in dims:
            if d > 65535:
                raise EmitterError("dimension %d does not fit pTensor's uint16 shapes" % d)
        if len(dims) > 4:
            raise EmitterError("rank-%d tensors (pTensor has at most 4 dims)" % len(dims))
        return "TensorShape(%s)" % ", ".join(str(int(d)) for d in dims)

    def handle(self, name):
        return "t%d" % self.act_index[name]

    def qparams_stmt(self, var, name):
        t = self.g.tensors[name]
        if t.dtype != INT8:
            return ""
        q = t.quant
        return "%s->set_quantization_params(PerTensorQuantizationParams(%d, %s));" % (
            var, int(q.zero_point[0]), _fmt_float(q.scale[0]))

    def create(self, name, lines):
        t = self.g.tensors[name]
        h = self.handle(name)
        shape = self.shape(t.shape)
        if self.plan is not None:
            lines.append("  %s = new BufferTensor(%s, %s, g_arena + %d);"
                         % (h, shape, _TTYPE[t.dtype], self.plan.offsets[name]))
        else:
            self.includes.add("pTensor/tensors/RamTensor.hpp")
            lines.append("  %s = new RamTensor(%s, %s);" % (h, shape, _TTYPE[t.dtype]))
        q = self.qparams_stmt(h, name)
        if q:
            lines.append("  " + q)

    def view2d(self, name):
        """Expression for a [1, N] view of activation `name` (dense input)."""
        t = self.g.tensors[name]
        if len(t.shape) == 2:
            return self.handle(name), ""
        stmt = "Tensor xv = new BufferTensor(%s, %s, %s->raw_mutable_data());" % (
            self.shape((1, t.size)), _TTYPE[t.dtype], self.handle(name))
        q = self.qparams_stmt("xv", name)
        return "xv", "    " + stmt + ("\n    " + q if q else "")

    # -- operators -------------------------------------------------------------

    def fused_act(self, act):
        lo, hi = clamp_range(act)
        named = {(-np.inf, np.inf): "none", (0.0, np.inf): "relu", (0.0, 6.0): "relu6",
                 (-1.0, 1.0): "relu1"}
        if (lo, hi) in named:
            return "FusedActivation::%s()" % named[(lo, hi)]
        return "FusedActivation::clamp(%s, %s)" % (
            _fmt_float(lo) if np.isfinite(lo) else "-INFINITY",
            _fmt_float(hi) if np.isfinite(hi) else "INFINITY")

    def tflm_act(self, node):
        act = node.attrs.get("activation")
        if act not in _TFLM_ACT:
            raise EmitterError("int8 %s with clamp %s" % (node.name, act))
        return "static_cast<TFLM::TfLiteFusedActivation>(%d)" % _TFLM_ACT[act]

    def node(self, node):
        """C++ statements (one block) for a node, outputs already created."""
        g = self.g
        op, a = node.op, node.attrs
        x = node.inputs[0] if node.inputs else None
        y = node.outputs[0]
        int8 = g.tensors[y].dtype == INT8 or (x is not None and g.tensors[x].dtype == INT8)
        X, Y = (self.handle(x) if x else None), self.handle(y)
        pre, body = [], None
        pad = a.get("padding")

        if op in LINEAR_OPS:
            w, b = node.inputs[1], node.inputs[2]
            pre.append("    Tensor w = %s;" % self.const(w))
            pre.append("    Tensor b = %s;" % self.const(b))
            sh, sw = a.get("stride", (1, 1))
            dh, dw = a.get("dilation", (1, 1))
            if int8:
                self.features.add("PTENSOR_WITH_QUANTIZED_OPS")
                n_out = g.tensors[w].shape[{"conv2d": 0, "depthwise_conv2d": 3, "dense": 1}[op]]
                pre.append("    " + self.weight_quant(w, n_out))
                self.max_scratch = max(self.max_scratch, 8 * n_out)
                if op == "dense":
                    self.includes.add("pTensor/ops/tflm/fully_connected.hpp")
                    xv, stmt = self.view2d(x)
                    if stmt:
                        pre.append(stmt)
                    body = ("typedef TflmSymQuantOps::FullyConnectedOperator<int8_t> Op;\n"
                            "    Op op(%s);\n"
                            "    op.set_inputs({{Op::input, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::output, %s}}).eval();" % (self.tflm_act(node), xv, Y))
                else:
                    conv = op == "conv2d"
                    self.includes.add("pTensor/ops/tflm/convolution2d.hpp" if conv else
                                      "pTensor/ops/tflm/depthwise_separable_convolution.hpp")
                    cls = "Conv2dOperator" if conv else "DepthwiseSeparableConvOperator"
                    args = ("strides, %s, dilation, %s" % (pad, self.tflm_act(node)) if conv else
                            "strides, %s, %d, dilation, %s" % (pad, a.get("depth_multiplier", 1),
                                                                self.tflm_act(node)))
                    body = ("const uint16_t strides[2] = {%d, %d};\n"
                            "    const uint16_t dilation[2] = {%d, %d};\n"
                            "    typedef TflmSymQuantOps::%s<int8_t> Op;\n"
                            "    Op op(%s);\n"
                            "    op.set_inputs({{Op::in, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::out, %s}}).eval();"
                            % (sh, sw, dh, dw, cls, args, X, Y))
            elif self.fused:
                self.features.add("PTENSOR_WITH_FUSION")
                self.includes.add("pTensor/ops/fused/FusedOperators.hpp")
                act = self.fused_act(a.get("activation"))
                if op == "conv2d":
                    body = ("typedef FusedConv2dOperator<float> Op;\n"
                            "    Op op(%d, %d, %s, %s, %d, %d);\n"
                            "    op.set_inputs({{Op::in, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::out, %s}}).eval();" % (sh, sw, pad, act, dh, dw, X, Y))
                elif op == "depthwise_conv2d":
                    body = ("typedef FusedDepthwiseConv2dOperator<float> Op;\n"
                            "    Op op(%d, %d, %s, %s, %d, %d, %d);\n"
                            "    op.set_inputs({{Op::in, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::out, %s}}).eval();"
                            % (sh, sw, pad, act, a.get("depth_multiplier", 1), dh, dw, X, Y))
                else:
                    body = ("typedef FusedFullyConnectedOperator<float> Op;\n"
                            "    Op op(%s);\n"
                            "    op.set_inputs({{Op::input, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::output, %s}}).eval();" % (act, X, Y))
            else:
                if a.get("activation") is not None:
                    raise EmitterError("%s has a fused activation but fusion is off" % node.name)
                if op == "conv2d":
                    if (dh, dw) != (1, 1):
                        raise EmitterError("dilated conv2d %s needs fuse=True (the reference "
                                           "Conv2dOperator has no dilation)" % node.name)
                    self.includes.add("pTensor/ops/reference/ConvolutionOperator.hpp")
                    body = ("typedef ReferenceOperators::Conv2dOperator<float> Op;\n"
                            "    Op op({%d, %d}, %s);\n"
                            "    op.set_inputs({{Op::in, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::out, %s}}).eval();" % (sh, sw, pad, X, Y))
                elif op == "depthwise_conv2d":
                    self.includes.add("pTensor/ops/reference/LayerOperators.hpp")
                    body = ("typedef DepthwiseConv2dOperator<float> Op;\n"
                            "    Op op(%d, %d, %s, %d, %d, %d);\n"
                            "    op.set_inputs({{Op::in, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::out, %s}}).eval();"
                            % (sh, sw, pad, a.get("depth_multiplier", 1), dh, dw, X, Y))
                else:
                    self.includes.add("pTensor/ops/reference/FullyConnectedOperator.hpp")
                    xv, stmt = self.view2d(x)
                    if stmt:
                        pre.append(stmt)
                    body = ("typedef FullyConnectedOperator<float> Op;\n"
                            "    Op op;\n"
                            "    op.set_inputs({{Op::input, %s}, {Op::filter, w}, {Op::bias, b}})"
                            ".set_outputs({{Op::output, %s}}).eval();" % (xv, Y))

        elif op in ("max_pool2d", "avg_pool2d"):
            if int8 and op == "avg_pool2d":
                raise EmitterError("int8 avg_pool2d has no TFLite-exact pTensor kernel")
            T = "int8_t" if int8 else "float"
            hdr, cls = (("MaxPoolOperator.hpp", "MaxPoolOperator") if op == "max_pool2d" else
                        ("AvgPoolOperator.hpp", "AvgPoolOperator"))
            self.includes.add("pTensor/ops/reference/" + hdr)
            (kh, kw), (sh, sw) = a["kernel"], a["stride"]
            body = ("typedef %s<%s> Op;\n    Op op({%d, %d}, {1, %d, %d, 1}, %s);\n"
                    "    op.set_inputs({{Op::in, %s}}).set_outputs({{Op::out, %s}}).eval();"
                    % (cls, T, kh, kw, sh, sw, pad, X, Y))

        elif op == "global_avg_pool":
            if int8:
                raise EmitterError("int8 global_avg_pool is not supported")
            self.includes.add("pTensor/ops/reference/LayerOperators.hpp")
            body = ("typedef GlobalAveragePoolOperator<float> Op;\n    Op op;\n"
                    "    op.set_inputs({{Op::in, %s}}).set_outputs({{Op::out, %s}}).eval();" % (X, Y))

        elif op == "batch_norm":
            if int8:
                raise EmitterError("int8 batch_norm (fold it with fuse=True before quantising)")
            self.includes.add("pTensor/ops/reference/LayerOperators.hpp")
            for i, nm in enumerate(("m", "v", "ga", "be")):
                pre.append("    Tensor %s = %s;" % (nm, self.const(node.inputs[i + 1])))
            body = ("typedef BatchNormOperator<float> Op;\n    Op op(%s);\n"
                    "    op.set_inputs({{Op::in, %s}, {Op::mean, m}, {Op::variance, v}, "
                    "{Op::gamma, ga}, {Op::beta, be}}).set_outputs({{Op::out, %s}}).eval();"
                    % (_fmt_float(a.get("epsilon", 1e-3)), X, Y))

        elif op == "activation":
            k = a["kind"]
            T = "int8_t" if int8 else "float"
            if k in ("relu", "relu6"):
                self.includes.add("pTensor/ops/reference/ReLuOperator.hpp")
                cls = "ReLUOperator" if k == "relu" else "ReLU6Operator"
                body = ("typedef %s<%s> Op;\n    Op op;\n"
                        "    op.set_inputs({{Op::in, %s}}).set_outputs({{Op::out, %s}}).eval();"
                        % (cls, T, X, Y))
            elif int8:
                raise EmitterError("int8 %s activation (%s) is not supported" % (k, node.name))
            elif k in ("relu1", "clip"):
                lo, hi = (-1.0, 1.0) if k == "relu1" else (a["lo"], a["hi"])
                self.includes.add("pTensor/ops/reference/ElementwiseOperators.hpp")
                body = ("ClipOperator op(activations::Clip(%s, %s));\n"
                        "    op.set_inputs({{ClipOperator::in, %s}}).set_outputs({{ClipOperator::out, %s}}).eval();"
                        % (_fmt_float(lo), _fmt_float(hi), X, Y))
            elif k == "sigmoid":
                self.includes.add("pTensor/ops/reference/LogisticOperator.hpp")
                body = ("typedef LogisticOperator<float> Op;\n    Op op;\n"
                        "    op.set_inputs({{Op::in, %s}}).set_outputs({{Op::out, %s}}).eval();" % (X, Y))
            elif k == "tanh":
                self.includes.add("pTensor/ops/reference/TanhOperator.hpp")
                body = ("typedef TanhOperator<float> Op;\n    Op op;\n"
                        "    op.set_inputs({{Op::act_in, %s}}).set_outputs({{Op::act_out, %s}}).eval();"
                        % (X, Y))
            else:
                raise EmitterError("activation %s" % k)

        elif op == "add":
            B = self.handle(node.inputs[1]) if not g.tensors[node.inputs[1]].is_const else "bconst"
            if B == "bconst":
                pre.append("    Tensor bconst = %s;" % self.const(node.inputs[1]))
            if int8:
                self.features.update(["PTENSOR_WITH_FUSION", "PTENSOR_WITH_QUANTIZED_OPS"])
                self.includes.add("pTensor/ops/fused/QuantizedAddOperator.hpp")
                body = ("typedef TflmSymQuantOps::QuantizedAddOperator<int8_t> Op;\n"
                        "    Op op(%s);\n"
                        "    op.set_inputs({{Op::a, %s}, {Op::b, %s}}).set_outputs({{Op::c, %s}}).eval();"
                        % (self.tflm_act(node), X, B, Y))
            elif self.fused:
                self.features.add("PTENSOR_WITH_FUSION")
                self.includes.add("pTensor/ops/fused/FusedOperators.hpp")
                body = ("typedef FusedAddOperator<float> Op;\n    Op op(%s);\n"
                        "    op.set_inputs({{Op::a, %s}, {Op::b, %s}}).set_outputs({{Op::c, %s}}).eval();"
                        % (self.fused_act(a.get("activation")), X, B, Y))
            else:
                if a.get("activation") is not None:
                    raise EmitterError("%s has a fused activation but fusion is off" % node.name)
                self.includes.add("pTensor/ops/reference/ElementwiseOperators.hpp")
                body = ("typedef BroadcastAddOperator<float> Op;\n    Op op;\n"
                        "    op.set_inputs({{Op::a, %s}, {Op::b, %s}}).set_outputs({{Op::c, %s}}).eval();"
                        % (X, B, Y))

        elif op == "softmax":
            if abs(a.get("beta", 1.0) - 1.0) > 1e-6:
                raise EmitterError("softmax with beta != 1")
            self.includes.add("pTensor/ops/reference/SoftMaxOperator.hpp")
            body = ("typedef SoftmaxOperator<%s> Op;\n    Op op;\n"
                    "    op.set_inputs({{Op::in, %s}}).set_outputs({{Op::out, %s}}).eval();"
                    % ("int8_t" if int8 else "float", X, Y))

        elif op == "quantize":
            self.features.add("PTENSOR_WITH_QUANTIZED_OPS")
            self.includes.add("pTensor/ops/tflm/QuantizeOperator.hpp")
            body = ("typedef TflmSymQuantOps::QuantizeOperator<int8_t, float> Op;\n    Op op;\n"
                    "    op.set_inputs({{Op::input, %s}}).set_outputs({{Op::output, %s}}).eval();"
                    % (X, Y))

        elif op == "dequantize":
            self.features.add("PTENSOR_WITH_QUANTIZED_OPS")
            self.includes.add("pTensor/ops/tflm/DequantizeOperator.hpp")
            body = ("typedef TflmSymQuantOps::DequantizeOperator<float, int8_t> Op;\n    Op op;\n"
                    "    op.set_inputs({{Op::a, %s}}).set_outputs({{Op::b, %s}}).eval();" % (X, Y))

        elif op == "reshape":
            return None  # handled by the caller (a view, or a copy for RamTensors)

        else:
            raise EmitterError("op %s has no pTensor emitter" % op)
        return "\n".join(pre + ["    " + body])

    # -- files -----------------------------------------------------------------

    def emit(self):
        g = self.g
        acts = g.activations()
        self.act_index = {n: i for i, n in enumerate(acts)}
        first_use, last_use = {}, {}
        for i, node in enumerate(g.nodes):
            for t in node.inputs + node.outputs:
                if t in self.act_index:
                    first_use.setdefault(t, i)
                    last_use[t] = i
        n_nodes = len(g.nodes)
        for t in g.outputs:
            last_use[t] = n_nodes
        for t in g.inputs:
            first_use[t] = -1

        body = []
        # graph inputs
        for k, name in enumerate(g.inputs):
            self.create(name, body)
            body.append("  memcpy(%s->raw_mutable_data(), in%d, %d);"
                        % (self.handle(name), k, g.tensors[name].nbytes))
        live = set(g.inputs)
        max_live = len(live)
        for i, node in enumerate(g.nodes):
            y = node.outputs[0]
            desc = node.op
            if node.attrs.get("activation") is not None:
                desc += ", %s" % (node.attrs["activation"],)
            if node.op == "activation":
                desc += " " + node.attrs["kind"]
            enc = [g.tensors[t].encoding.describe() for t in node.inputs
                   if g.tensors[t].encoding is not None]
            if enc:
                desc += ", weights " + enc[0]
            body.append("  // %d: %s (%s) -> %s" % (i, node.name, desc, list(g.tensors[y].shape)))
            if node.op == "reshape":
                x = node.inputs[0]
                if self.plan is not None:
                    self.create(y, body)
                else:
                    self.create(y, body)
                    body.append("  memcpy(%s->raw_mutable_data(), %s->raw_data(), %d);"
                                % (self.handle(y), self.handle(x), g.tensors[y].nbytes))
            else:
                self.create(y, body)
                body.append("  {\n%s\n  }" % self.node(node))
            live.add(y)
            max_live = max(max_live, len(live))
            body.append("  if (g_err.failed) return -2;")
            for t in list(node.inputs) + [y]:
                if t in live and last_use.get(t) == i:
                    body.append("  %s.free();" % self.handle(t))
                    live.discard(t)
        for k, name in enumerate(g.outputs):
            body.append("  memcpy(out%d, %s->raw_data(), %d);" % (k, self.handle(name),
                                                                 g.tensors[name].nbytes))
        for name in sorted(live, key=lambda n: self.act_index[n]):
            body.append("  %s.free();" % self.handle(name))

        if self.fast_kernels:
            self.features.add("PTENSOR_WITH_FAST_KERNELS")
        if g.is_quantized():
            self.includes.add("pTensor/core/quantizationPrimitives.hpp")
        return {
            self.name + ".h": self.header(),
            self.name + ".cpp": self.source(body, max_live, acts),
            self.name + ".cmake": self.cmake(),
        }

    def io_params(self):
        g = self.g
        params, macros = [], []
        for prefix, names, const in (("IN", g.inputs, "const "), ("OUT", g.outputs, "")):
            for k, n in enumerate(names):
                t = g.tensors[n]
                params.append("%s%s* %s%d" % (const, _CTYPE[t.dtype], prefix.lower(), k))
                macros.append("#define %s_%s%d_SIZE %d  /* %s %s */"
                              % (self.NAME, prefix, k, t.size, t.dtype, list(t.shape)))
                if t.dtype == INT8:
                    macros.append("#define %s_%s%d_SCALE %s" % (self.NAME, prefix, k, _fmt_float(t.quant.scale[0])))
                    macros.append("#define %s_%s%d_ZERO_POINT (%d)" % (self.NAME, prefix, k, int(t.quant.zero_point[0])))
        return params, macros

    def header(self):
        params, macros = self.io_params()
        arena = self.plan.arena_bytes if self.plan is not None else 0
        return """/* Generated by ptensor.codegen. Do not edit. */
#ifndef {N}_H
#define {N}_H

#include <stddef.h>
#include <stdint.h>

{macros}
#define {N}_ARENA_BYTES {arena}  /* planned activation arena (0: dynamic RamTensors) */
#define {N}_WEIGHT_BYTES {flash}  /* constant data (flash) */

#ifdef __cplusplus
extern "C" {{
#endif

/* Installs the model's allocators and error handler on pTensor's default
 * context and decodes any weights kept in RAM. Called by {n}_invoke() if
 * needed. Returns 0 on success. */
int {n}_init(void);

/* Runs the model once (batch 1). Returns 0 on success, nonzero if pTensor
 * raised an error. */
int {n}_invoke({params});

/* Peak bytes used in the metadata and RAM arenas so far (for tuning
 * {N}_META_ARENA_BYTES / {N}_RAM_ARENA_BYTES). */
size_t {n}_meta_high_water(void);
size_t {n}_ram_high_water(void);

#ifdef __cplusplus
}}
#endif

#endif
""".format(N=self.NAME, n=self.name, macros="\n".join(macros), arena=arena,
           flash=self.flash_bytes, params=", ".join(params))

    def source(self, body, max_live, acts):
        g = self.g
        n_handles = max_live + 6
        meta = self.meta_arena_bytes or (2048 + 192 * n_handles)
        blocks = 4 * n_handles + 16
        if self.plan is not None:
            ram = self.ram_arena_bytes or (512 + self.max_scratch + 64)
        else:
            naive = sum(g.tensors[n].nbytes + 16 for n in acts)
            ram = self.ram_arena_bytes or (naive + self.max_scratch + 1024)
        params, _ = self.io_params()
        feature_checks = []
        for f in sorted(self.features):
            if f == "PTENSOR_WITH_FAST_KERNELS":
                continue
            feature_checks.append('#if !%s\n#error "%s needs pTensor built with %s=ON (see %s.cmake)"\n#endif'
                                  % (f, self.name, f, self.name))
        includes = "\n".join('#include "%s"' % h for h in sorted(self.includes))
        handles = "\n".join("  Tensor %s;  // %s %s %s" % (self.handle(n), n, list(g.tensors[n].shape),
                                                         g.tensors[n].dtype) for n in acts)
        arena = ("alignas(8) uint8_t g_arena[%d];  // %s\n" % (max(self.plan.arena_bytes, 8),
                                                              self.plan.describe())
                 if self.plan is not None else "")
        init_extra = "\n".join(self.init_lines)
        graph_summary = "\n".join(" * " + line for line in g.summary().splitlines())
        notes = "\n".join(" *   " + n for n in g.notes)
        return """/* Generated by ptensor.codegen. Do not edit.
 *
{summary}
 *
 * Options: {opts}
 * Passes:
{notes}
 */
#include "{n}.h"

#include <math.h>
#include <string.h>

#include "pTensor/ptensor_config.h"
{includes}

{checks}

#ifndef {N}_WEIGHT_ATTR
#define {N}_WEIGHT_ATTR /* e.g. __attribute__((section(".rodata.model"))) */
#endif
#ifndef {N}_META_ARENA_BYTES
#define {N}_META_ARENA_BYTES {meta}
#endif
#ifndef {N}_RAM_ARENA_BYTES
#define {N}_RAM_ARENA_BYTES {ram}
#endif

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {{

{decls}

{arena}#ifndef {N}_EXTERNAL_CONTEXT
localCircularArenaAllocator<{N}_META_ARENA_BYTES, uint32_t, {blocks}> g_meta;
localCircularArenaAllocator<{N}_RAM_ARENA_BYTES, uint32_t, {blocks}> g_ram;
#endif

class FlagErrorHandler : public ErrorHandler {{
 public:
  volatile int failed = 0;
  virtual void uThrow(Error*) override {{ failed = 1; }}
  virtual void notify(const Event&) override {{}}
}};
FlagErrorHandler g_err;
bool g_ready = false;

}}  // namespace

extern "C" int {n}_init(void) {{
  Context* ctx = Context::get_default_context();
#ifndef {N}_EXTERNAL_CONTEXT
  ctx->set_metadata_allocator(&g_meta);
  ctx->set_ram_data_allocator(&g_ram);
#endif
  ctx->set_ErrorHandler(&g_err);
  g_err.failed = 0;
{init_extra}
  g_ready = !g_err.failed;
  return g_ready ? 0 : -1;
}}

extern "C" size_t {n}_meta_high_water(void) {{
#ifndef {N}_EXTERNAL_CONTEXT
  return g_meta.high_water_mark();
#else
  return 0;
#endif
}}

extern "C" size_t {n}_ram_high_water(void) {{
#ifndef {N}_EXTERNAL_CONTEXT
  return g_ram.high_water_mark();
#else
  return 0;
#endif
}}

extern "C" int {n}_invoke({params}) {{
  if (!g_ready && {n}_init() != 0) return -1;
  g_err.failed = 0;
{handles}
{body}
  return g_err.failed ? -2 : 0;
}}
""".format(summary=graph_summary, opts=self.options_comment, notes=notes, n=self.name,
           N=self.NAME, includes=includes, checks="\n".join(feature_checks), meta=meta, ram=ram,
           blocks=blocks, decls="\n\n".join(self.decls), arena=arena, init_extra=init_extra,
           params=", ".join(params), handles=handles, body="\n".join(body))

    def cmake(self):
        lines = ["# Generated by ptensor.codegen: the pTensor features model '%s' needs." % self.name,
                 "# include() this file before add_subdirectory(<ptensor>), then link %s." % self.name,
                 "# Features the model needs are forced ON; the others default to OFF unless",
                 "# something else in the project already set them (combine models freely).", ""]
        for f in FEATURES:
            if f in self.features:
                lines.append('set(%s ON CACHE BOOL "needed by %s" FORCE)' % (f, self.name))
            else:
                lines.append("if(NOT DEFINED %s)\n  set(%s OFF CACHE BOOL \"not needed by %s\")\nendif()"
                             % (f, f, self.name))
        lines += ["",
                  "add_library(%s STATIC ${CMAKE_CURRENT_LIST_DIR}/%s.cpp)" % (self.name, self.name),
                  "target_include_directories(%s PUBLIC ${CMAKE_CURRENT_LIST_DIR})" % self.name,
                  "target_link_libraries(%s PUBLIC ptensor)" % self.name,
                  "# pTensor is built without RTTI; the model must match.",
                  'if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")',
                  "  target_compile_options(%s PRIVATE -fno-rtti)" % self.name,
                  "elseif(MSVC)",
                  "  target_compile_options(%s PRIVATE /GR-)" % self.name,
                  "endif()", ""]
        return "\n".join(lines)
