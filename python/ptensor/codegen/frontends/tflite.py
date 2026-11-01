"""TFLite flatbuffers (float32, or full-integer int8 as the TFLite converter
writes them). Needs the ``tflite`` package (``pip install tflite``), a
flatbuffer reader with no TensorFlow dependency."""
import numpy as np

from ..ir import FLOAT32, INT8, INT32, Graph, QuantParams
from .base import Frontend, FrontendUnavailable, register_frontend

_ACT = {0: None, 1: "relu", 2: "relu1", 3: "relu6"}
_PAD = {0: "SAME", 1: "VALID"}


def _tfl():
    try:
        import tflite
    except ImportError:
        raise FrontendUnavailable("the TFLite frontend needs the 'tflite' package: pip install tflite")
    return tflite


def _short(name):
    """TFLite names concatenate every fused op's name with ';'. Keep the first."""
    return name.split(";")[0].split("/")[-2] if "/" in name.split(";")[0] else name.split(";")[0]


@register_frontend
class TFLiteFrontend(Frontend):
    name = "tflite"
    extensions = (".tflite",)

    def accepts(self, source):
        if isinstance(source, (bytes, bytearray)):
            return bytes(source[4:8]) == b"TFL3"
        return Frontend.accepts(self, source)

    def load(self, source, name=None, **options):
        if isinstance(source, str):
            import os
            name = name or os.path.splitext(os.path.basename(source))[0]
            with open(source, "rb") as f:
                source = f.read()
        return _Importer(bytes(source), name or "model").run()


class _Importer(object):
    def __init__(self, buf, name):
        self.t = _tfl()
        self.buf = buf
        self.model = self.t.Model.GetRootAsModel(buf, 0)
        if self.model.SubgraphsLength() != 1:
            raise NotImplementedError("only single-subgraph TFLite models are supported")
        self.sg = self.model.Subgraphs(0)
        self.g = Graph(name)
        self.g.source = "tflite"
        self.names = {}  # tflite tensor index -> IR name
        tt = self.t.TensorType
        self.dtypes = {tt.FLOAT32: FLOAT32, tt.INT8: INT8, tt.INT32: INT32}
        self.op_names = {v: k for k, v in vars(self.t.BuiltinOperator).items()
                         if not k.startswith("_")}

    # -- tensors --------------------------------------------------------------

    def raw(self, i):
        t = self.sg.Tensors(i)
        shape = tuple(int(d) for d in t.ShapeAsNumpy()) if t.ShapeLength() else ()
        buf = self.model.Buffers(t.Buffer())
        data = None
        if buf is not None and buf.DataLength():
            data = buf.DataAsNumpy().tobytes()
        elif buf is not None and hasattr(buf, "Offset") and buf.Offset() > 1:
            data = self.buf[buf.Offset(): buf.Offset() + buf.Size()]
        q = t.Quantization()
        quant = None
        if q is not None and q.ScaleLength():
            scale = q.ScaleAsNumpy().astype(np.float32)
            zp = q.ZeroPointAsNumpy().astype(np.int32) if q.ZeroPointLength() else np.zeros_like(scale, np.int32)
            quant = QuantParams(scale, zp, q.QuantizedDimension() if scale.size > 1 else None)
        return t, shape, data, quant

    def tensor(self, i, transform=None):
        """IR name of TFLite tensor i (created on first use)."""
        if i in self.names and transform is None:
            return self.names[i]
        t, shape, data, quant = self.raw(i)
        if t.Type() not in self.dtypes:
            if t.Type() == self.t.TensorType.FLOAT16 and data is not None:
                arr = np.frombuffer(data, "<f2").astype(np.float32).reshape(shape)
                name = self.g.add_const(_short(t.Name().decode()), arr).name
                self.names[i] = name
                return name
            raise NotImplementedError("tensor %s has unsupported TFLite type %d"
                                      % (t.Name().decode(), t.Type()))
        dtype = self.dtypes[t.Type()]
        arr = None
        if data is not None:
            arr = np.frombuffer(data, np.dtype(dtype).newbyteorder("<")).astype(dtype).reshape(shape)
        if transform is not None:
            arr, shape, quant = transform(arr, shape, quant)
        shape = shape if shape else (1,)
        if arr is None and len(shape) == 0:
            shape = (1,)
        name = self.g.add_tensor(_short(t.Name().decode()), shape, dtype, arr, quant).name
        if transform is None:
            self.names[i] = name
        return name

    def zero_bias(self, base, n, x_idx, w_quant):
        x_quant = self.raw(x_idx)[3]
        if x_quant is None:
            return self.g.add_const(base + "_bias", np.zeros(n, np.float32)).name
        scale = w_quant.scale * x_quant.scale[0] if w_quant is not None else x_quant.scale
        return self.g.add_const(base + "_bias", np.zeros(n, np.int32),
                                quant=QuantParams(scale, 0, 0 if np.size(scale) > 1 else None)).name

    def opts(self, op, cls):
        table = op.BuiltinOptions()
        o = cls()
        if table is not None:
            o.Init(table.Bytes, table.Pos)
        return o

    # -- the walk ------------------------------------------------------------

    def run(self):
        sg = self.sg
        for i in sg.InputsAsNumpy():
            self.g.inputs.append(self.tensor(int(i)))
        for k in range(sg.OperatorsLength()):
            op = sg.Operators(k)
            code = self.model.OperatorCodes(op.OpcodeIndex())
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            self.operator(op, builtin, k)
        for i in sg.OutputsAsNumpy():
            self.g.outputs.append(self.tensor(int(i)))
        self.g.remove_unused()
        return self.g

    def operator(self, op, builtin, k):
        B = self.t.BuiltinOperator
        g = self.g
        ins = [int(i) for i in op.InputsAsNumpy()]
        outs = [int(i) for i in op.OutputsAsNumpy()]
        opname = self.op_names.get(builtin, str(builtin))
        nm = "%s_%d" % (opname.lower(), k)

        # Shape arithmetic (SHAPE -> STRIDED_SLICE -> PACK feeding a RESHAPE,
        # which Keras Flatten produces with a dynamic batch). Shapes are
        # static here and RESHAPE takes its output shape from the output
        # tensor, so these ops are dropped; anything else reading their
        # results fails graph validation with a clear message.
        int_types = (self.t.TensorType.INT32, self.t.TensorType.INT64)
        if builtin in (B.SHAPE, B.STRIDED_SLICE, B.PACK, B.CONCATENATION, B.GATHER, B.CAST) \
                and all(self.sg.Tensors(o).Type() in int_types for o in outs):
            return

        if builtin in (B.CONV_2D, B.DEPTHWISE_CONV_2D):
            conv = builtin == B.CONV_2D
            o = self.opts(op, self.t.Conv2DOptions if conv else self.t.DepthwiseConv2DOptions)
            x = self.tensor(ins[0])
            w = self.tensor(ins[1])
            wt = g.tensors[w]
            n_out = wt.shape[0] if conv else wt.shape[3]
            b = self.tensor(ins[2]) if len(ins) > 2 and ins[2] >= 0 else \
                self.zero_bias(nm, n_out, ins[0], wt.quant)
            attrs = {"stride": (o.StrideH(), o.StrideW()), "padding": _PAD[o.Padding()],
                     "dilation": (o.DilationHFactor(), o.DilationWFactor()),
                     "activation": _ACT[o.FusedActivationFunction()]}
            if not conv:
                xin = g.tensors[x].shape[3]
                attrs["depth_multiplier"] = n_out // xin
            g.add_node("conv2d" if conv else "depthwise_conv2d", [x, w, b], [self.tensor(outs[0])],
                       attrs, nm)
            return

        if builtin == B.FULLY_CONNECTED:
            o = self.opts(op, self.t.FullyConnectedOptions)
            x = self.tensor(ins[0])

            def to_in_out(arr, shape, quant):  # TFLite [out, in] -> pTensor [in, out]
                if quant is not None and quant.per_channel:
                    quant.axis = 1
                return np.ascontiguousarray(arr.T), (shape[1], shape[0]), quant
            w = self.tensor(ins[1], to_in_out)
            n_out = g.tensors[w].shape[1]
            b = self.tensor(ins[2]) if len(ins) > 2 and ins[2] >= 0 else \
                self.zero_bias(nm, n_out, ins[0], g.tensors[w].quant)
            y = self.tensor(outs[0])
            if g.tensors[y].shape != (1, n_out):
                raise NotImplementedError("FULLY_CONNECTED %s with output %s" % (nm, g.tensors[y].shape))
            g.add_node("dense", [x, w, b], [y], {"activation": _ACT[o.FusedActivationFunction()]}, nm)
            return

        if builtin in (B.MAX_POOL_2D, B.AVERAGE_POOL_2D):
            o = self.opts(op, self.t.Pool2DOptions)
            x = self.tensor(ins[0])
            y = self.tensor(outs[0])
            act = _ACT[o.FusedActivationFunction()]
            attrs = {"kernel": (o.FilterHeight(), o.FilterWidth()),
                     "stride": (o.StrideH(), o.StrideW()), "padding": _PAD[o.Padding()]}
            kind = "max_pool2d" if builtin == B.MAX_POOL_2D else "avg_pool2d"
            if act is None:
                g.add_node(kind, [x], [y], attrs, nm)
            else:
                mid = g.add_tensor(nm, g.tensors[y].shape, g.tensors[y].dtype,
                                   quant=g.tensors[y].quant).name
                g.add_node(kind, [x], [mid], attrs, nm)
                g.add_node("activation", [mid], [y], {"kind": act}, nm + "_act")
            return

        if builtin in (B.RESHAPE, B.SQUEEZE):
            g.add_node("reshape", [self.tensor(ins[0])], [self.tensor(outs[0])], name=nm)
            return

        if builtin == B.SOFTMAX:
            o = self.opts(op, self.t.SoftmaxOptions)
            g.add_node("softmax", [self.tensor(ins[0])], [self.tensor(outs[0])],
                       {"beta": float(o.Beta())}, nm)
            return

        if builtin == B.ADD:
            o = self.opts(op, self.t.AddOptions)
            g.add_node("add", [self.tensor(ins[0]), self.tensor(ins[1])], [self.tensor(outs[0])],
                       {"activation": _ACT[o.FusedActivationFunction()]}, nm)
            return

        kinds = {B.RELU: "relu", B.RELU6: "relu6", B.LOGISTIC: "sigmoid", B.TANH: "tanh",
                 B.RELU_N1_TO_1: "relu1"}
        if builtin in kinds:
            g.add_node("activation", [self.tensor(ins[0])], [self.tensor(outs[0])],
                       {"kind": kinds[builtin]}, nm)
            return

        if builtin == B.MEAN:
            axis_data = self.raw(ins[1])[2]
            if axis_data is None:
                raise NotImplementedError("MEAN %s with non-constant axes" % nm)
            axes = set(int(a) % 4 for a in np.frombuffer(axis_data, "<i4"))
            if axes != {1, 2}:
                raise NotImplementedError("MEAN over axes %s (only global average pooling)" % axes)
            g.add_node("global_avg_pool", [self.tensor(ins[0])], [self.tensor(outs[0])], name=nm)
            return

        if builtin == B.QUANTIZE:
            g.add_node("quantize", [self.tensor(ins[0])], [self.tensor(outs[0])], name=nm)
            return

        if builtin == B.DEQUANTIZE:
            src = self.raw(ins[0])
            if src[2] is not None:  # a constant (e.g. float16 weights): decode it now
                self.names[outs[0]] = self.tensor(ins[0])
                return
            g.add_node("dequantize", [self.tensor(ins[0])], [self.tensor(outs[0])], name=nm)
            return

        raise NotImplementedError("TFLite operator %s has no pTensor mapping yet" % opname)
