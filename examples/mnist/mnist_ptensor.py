"""Classify MNIST digits with pTensor, running a TFLite model in its own dtype.

The sample reads a .tflite model, takes each tensor's data type and
quantization from the model, and runs every layer through the pTensor C++
kernels via the Python bindings. The default model, mnist_int8.tflite, is a
full-integer model (int8 activations, per-channel int8 weights, int32 biases),
which is how it would run on a microcontroller, so the whole network runs in
int8. Point --model at a float32 .tflite (such as mnist_f32.tflite) and the
same code runs it in float32 instead.

    # from the repository root
    cmake -S . -B build -DPTENSOR_BUILD_C_API=ON && cmake --build build
    pip install numpy tflite
    PYTHONPATH=python python3 examples/mnist/mnist_ptensor.py

If TensorFlow or ai-edge-litert is installed, the sample also runs the TFLite
interpreter on the same images and reports how closely pTensor matches it.
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import tflite
except ImportError:  # pragma: no cover
    sys.exit("This sample needs the 'tflite' package: pip install tflite")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
import ptensor  # noqa: E402

import mnist_data  # noqa: E402

DTYPES = {
    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.INT64: np.int64,
}
PADDING = {tflite.Padding.SAME: "SAME", tflite.Padding.VALID: "VALID"}
ACTIVATION = {
    tflite.ActivationFunctionType.NONE: None,
    tflite.ActivationFunctionType.RELU: "relu",
    tflite.ActivationFunctionType.RELU6: "relu6",
}
OP_NAMES = {v: k for k, v in vars(tflite.BuiltinOperator).items()
            if not k.startswith("_")}


class TensorInfo(object):
    """Name, shape, dtype, quantization and (for constants) data of a tensor."""

    def __init__(self, model, t):
        self.name = t.Name().decode()
        self.shape = tuple(int(d) for d in t.ShapeAsNumpy()) if t.ShapeLength() else ()
        if t.Type() not in DTYPES:
            raise NotImplementedError("tensor %s has unsupported type %d"
                                      % (self.name, t.Type()))
        self.dtype = DTYPES[t.Type()]
        q = t.Quantization()
        self.scale = None
        self.zero_point = None
        if q is not None and q.ScaleLength():
            self.scale = q.ScaleAsNumpy().astype(np.float32)
            self.zero_point = q.ZeroPointAsNumpy().astype(np.int32)
        buf = model.Buffers(t.Buffer())
        self.data = None
        if buf is not None and buf.DataLength():
            self.data = np.frombuffer(buf.DataAsNumpy().tobytes(),
                                      self.dtype).reshape(self.shape)

    @property
    def quantized(self):
        return self.scale is not None

    def qparams(self):
        """(scale, zero_point) for a per-tensor quantized activation."""
        return float(self.scale[0]), int(self.zero_point[0])

    def describe(self):
        s = "%s%s" % (np.dtype(self.dtype).name, list(self.shape))
        if self.quantized:
            if self.scale.size > 1:
                s += " per-channel x%d" % self.scale.size
            else:
                s += " scale=%.6g zp=%d" % self.qparams()
        return s


def _options(op, cls):
    table = op.BuiltinOptions()
    opts = cls()
    if table is not None:
        opts.Init(table.Bytes, table.Pos)
    return opts


class PTensorModel(object):
    """A minimal TFLite interpreter whose kernels are pTensor's."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.model = tflite.Model.GetRootAsModel(self.buf, 0)
        if self.model.SubgraphsLength() != 1:
            raise NotImplementedError("only single-subgraph models are supported")
        g = self.model.Subgraphs(0)
        self.tensors = [TensorInfo(self.model, g.Tensors(i))
                        for i in range(g.TensorsLength())]
        self.inputs = [int(i) for i in g.InputsAsNumpy()]
        self.outputs = [int(i) for i in g.OutputsAsNumpy()]
        self.ops = []
        for i in range(g.OperatorsLength()):
            op = g.Operators(i)
            code = self.model.OperatorCodes(op.OpcodeIndex())
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            self.ops.append(self._prepare(op, builtin))

    @property
    def input(self):
        return self.tensors[self.inputs[0]]

    @property
    def output(self):
        return self.tensors[self.outputs[0]]

    # -- build a closure per operator, with constants prepared once ----------

    def _prepare(self, op, builtin):
        ins = [int(i) for i in op.InputsAsNumpy()]
        outs = [int(i) for i in op.OutputsAsNumpy()]
        T = self.tensors
        x, y = T[ins[0]], T[outs[0]]
        name = OP_NAMES.get(builtin, str(builtin))
        int8 = x.dtype == np.int8
        bias = T[ins[2]].data if len(ins) > 2 and ins[2] >= 0 else None

        if builtin == tflite.BuiltinOperator.CONV_2D:
            o = _options(op, tflite.Conv2DOptions)
            if o.DilationHFactor() != 1 or o.DilationWFactor() != 1:
                raise NotImplementedError("dilated CONV_2D")
            w = T[ins[1]]
            stride = (o.StrideH(), o.StrideW())
            pad = PADDING[o.Padding()]
            act = ACTIVATION[o.FusedActivationFunction()]
            if int8:
                xs, xz = x.qparams()
                ys, yz = y.qparams()

                def run(v):
                    return ptensor.conv2d_int8(v, xs, xz, w.data, w.scale, bias,
                                               ys, yz, stride, pad, act)
            else:
                def run(v):
                    out = ptensor.conv2d(v, w.data, bias, stride, pad)
                    return _float_act(out, act)
            kind = "%dx%d conv, %d filters" % (w.shape[1], w.shape[2], w.shape[0])

        elif builtin in (tflite.BuiltinOperator.MAX_POOL_2D,
                         tflite.BuiltinOperator.AVERAGE_POOL_2D):
            o = _options(op, tflite.Pool2DOptions)
            k = (o.FilterHeight(), o.FilterWidth())
            stride = (o.StrideH(), o.StrideW())
            pad = PADDING[o.Padding()]
            act = ACTIVATION[o.FusedActivationFunction()]
            is_max = builtin == tflite.BuiltinOperator.MAX_POOL_2D
            if int8 and not is_max:
                raise NotImplementedError("int8 AVERAGE_POOL_2D")
            if int8:
                if act is not None:
                    raise NotImplementedError("fused activation on int8 pooling")

                def run(v):
                    return ptensor.maxpool2d_int8(v, k, stride, pad)
            else:
                fn = ptensor.maxpool2d if is_max else ptensor.avgpool2d

                def run(v):
                    return _float_act(fn(v, k, stride, pad), act)
            kind = "%dx%d %s pool" % (k[0], k[1], "max" if is_max else "avg")

        elif builtin == tflite.BuiltinOperator.RESHAPE:
            shape = (-1,) + y.shape[1:]

            def run(v):
                return v.reshape(shape)
            kind = "reshape"

        elif builtin == tflite.BuiltinOperator.FULLY_CONNECTED:
            o = _options(op, tflite.FullyConnectedOptions)
            w = T[ins[1]]
            act = ACTIVATION[o.FusedActivationFunction()]
            wt = np.ascontiguousarray(w.data.T)  # TFLite [out,in] -> pTensor [in,out]
            if int8:
                xs, xz = x.qparams()
                ys, yz = y.qparams()

                def run(v):
                    v = v.reshape(v.shape[0], -1)
                    return ptensor.fully_connected_int8(v, xs, xz, wt, w.scale, bias,
                                                        ys, yz, act)
            else:
                def run(v):
                    return ptensor.fully_connected(v.reshape(v.shape[0], -1), wt,
                                                   bias, act)
            kind = "dense %d -> %d" % (w.shape[1], w.shape[0])

        elif builtin == tflite.BuiltinOperator.SOFTMAX:
            o = _options(op, tflite.SoftmaxOptions)
            if abs(o.Beta() - 1.0) > 1e-6:
                raise NotImplementedError("SOFTMAX with beta != 1")
            if int8:
                xs, xz = x.qparams()
                ys, yz = y.qparams()

                def run(v):
                    return ptensor.softmax_int8(v, xs, xz, ys, yz)
            else:
                run = ptensor.softmax
            kind = "softmax"

        elif builtin == tflite.BuiltinOperator.QUANTIZE:
            ys, yz = y.qparams()

            def run(v):
                return ptensor.quantize(v, ys, yz)
            kind = "quantize"

        elif builtin == tflite.BuiltinOperator.DEQUANTIZE:
            xs, xz = x.qparams()

            def run(v):
                return ptensor.dequantize(v, xs, xz)
            kind = "dequantize"

        else:
            raise NotImplementedError("TFLite operator %s is not wired up in this "
                                      "sample" % name)
        return {"name": name, "kind": kind, "in": ins[0], "out": outs[0],
                "run": run, "dtype": np.dtype(y.dtype).name}

    def summary(self):
        lines = ["input  %s" % self.input.describe()]
        for op in self.ops:
            lines.append("  %-16s %-24s -> %s" % (op["name"], op["kind"],
                                                  self.tensors[op["out"]].describe()))
        lines.append("output %s" % self.output.describe())
        return "\n".join(lines)

    def prepare_input(self, images):
        """uint8 images [N,28,28] -> the model's input dtype and layout."""
        x = (images.astype(np.float32) / 255.0).reshape((-1,) + self.input.shape[1:])
        if self.input.dtype == np.int8:
            return ptensor.quantize(x, *self.input.qparams())
        return x

    def run(self, x):
        values = {self.inputs[0]: x}
        for op in self.ops:
            values[op["out"]] = op["run"](values[op["in"]])
        return values[self.outputs[0]]


def _float_act(v, act):
    if act == "relu":
        return ptensor.relu(v)
    if act == "relu6":
        return ptensor.relu6(v)
    return v


def tflite_reference(model_path):
    """Return a function running the TFLite interpreter, or None.

    It uses TFLite's reference kernels, which define the int8 arithmetic.
    The default XNNPACK delegate rounds requantization differently and can
    differ by 1 LSB.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        from ai_edge_litert import interpreter as litert
        Interpreter, resolver = litert.Interpreter, litert.OpResolverType
    except ImportError:
        try:
            import warnings
            warnings.filterwarnings("ignore", module="tensorflow")
            import tensorflow as tf
        except ImportError:
            return None
        Interpreter = tf.lite.Interpreter
        resolver = tf.lite.experimental.OpResolverType
    it = Interpreter(model_path=model_path,
                     experimental_op_resolver_type=resolver.BUILTIN_REF)
    it.allocate_tensors()
    i = it.get_input_details()[0]["index"]
    o = it.get_output_details()[0]["index"]

    def run(x):
        out = []
        for row in x:
            it.set_tensor(i, row[None])
            it.invoke()
            out.append(it.get_tensor(o)[0].copy())
        return np.stack(out)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.join(HERE, "mnist_int8.tflite"))
    ap.add_argument("-n", "--count", type=int, default=1000,
                    help="number of test images to classify (max 10000)")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the TFLite interpreter comparison")
    args = ap.parse_args()

    model = PTensorModel(args.model)
    print("pTensor %s, model %s" % (ptensor.__version__, os.path.basename(args.model)))
    print(model.summary())

    images, labels = mnist_data.load("test")
    images, labels = images[:args.count], labels[:args.count]
    x = model.prepare_input(images)

    t0 = time.perf_counter()
    out = np.concatenate([model.run(x[i:i + args.batch])
                          for i in range(0, len(x), args.batch)])
    dt = time.perf_counter() - t0
    pred = out.argmax(axis=1)
    acc = float((pred == labels).mean())
    print("\npTensor: %d images, accuracy %.2f%%, %.2f ms/image"
          % (len(x), 100 * acc, 1000 * dt / len(x)))
    print("first 10: predicted %s" % pred[:10].tolist())
    print("          actual    %s" % labels[:10].tolist())

    ref = None if args.no_reference else tflite_reference(args.model)
    if ref is not None:
        r = ref(x)
        r_pred = r.argmax(axis=1)
        diff = np.abs(out.astype(np.int32) - r.astype(np.int32)) \
            if out.dtype == np.int8 else np.abs(out - r)
        print("\nTFLite:  accuracy %.2f%%; pTensor picks the same digit for "
              "%.2f%% of images" % (100 * (r_pred == labels).mean(),
                                    100 * (r_pred == pred).mean()))
        unit = "LSB" if out.dtype == np.int8 else ""
        print("         output max |diff| %s %s, mean %.4f" % (diff.max(), unit, diff.mean()))
    return 0 if acc > 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
