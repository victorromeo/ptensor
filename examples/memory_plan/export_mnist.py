"""Convert examples/mnist/mnist_int8.tflite to the .ptm format, and write the
MNIST test set, for the memory_plan runner.

    pip install numpy tflite
    python3 examples/memory_plan/export_mnist.py out/

Writes out/mnist_int8.ptm and out/mnist_test.bin. The .ptm layout is the one
described in python/ptensor/ptm.py (kind 1, int8 activations); the TFLite
RESHAPE before the first dense layer and the final SOFTMAX are dropped (the
runner reads the pooled tensor as [1, 784] and takes the argmax of the
logits, which softmax does not change).
"""
import os
import struct
import sys

import numpy as np

try:
    import tflite
except ImportError:  # pragma: no cover
    sys.exit("This script needs the 'tflite' package: pip install tflite")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mnist"))
import mnist_data  # noqa: E402

OP_CONV, OP_POOL, OP_DENSE = 1, 2, 3
ST_I8, ST_I32 = 3, 4
PAD = {tflite.Padding.SAME: 2, tflite.Padding.VALID: 1}
ACT = {tflite.ActivationFunctionType.NONE: 0,
       tflite.ActivationFunctionType.RELU: 1,
       tflite.ActivationFunctionType.RELU6: 3}
B = tflite.BuiltinOperator


def tensor_record(a):
    st = {np.dtype(np.int8): ST_I8, np.dtype(np.int32): ST_I32}[a.dtype]
    data = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
    dims = struct.pack("<B", a.ndim) + b"".join(struct.pack("<H", d) for d in a.shape)
    return struct.pack("<B", st) + dims + struct.pack("<I", len(data)) + data


def options(op, cls):
    o = cls()
    t = op.BuiltinOptions()
    o.Init(t.Bytes, t.Pos)
    return o


def convert(path):
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)

    def tensor(i):
        t = g.Tensors(i)
        q = t.Quantization()
        dt = {tflite.TensorType.INT8: np.int8, tflite.TensorType.INT32: np.int32}[t.Type()]
        data = m.Buffers(t.Buffer()).DataAsNumpy()
        shape = tuple(int(d) for d in t.ShapeAsNumpy())
        arr = None
        if not isinstance(data, int):
            arr = np.frombuffer(data.tobytes(), dt).reshape(shape)
        return arr, shape, q.ScaleAsNumpy().astype(np.float32), q.ZeroPointAsNumpy()

    _, in_shape, in_s, in_z = tensor(int(g.InputsAsNumpy()[0]))
    layers = []
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        code = m.OperatorCodes(op.OpcodeIndex())
        b = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        ins, outs = op.InputsAsNumpy(), op.OutputsAsNumpy()
        if b in (B.CONV_2D, B.FULLY_CONNECTED):
            w, _, w_s, _ = tensor(ins[1])
            bias = tensor(ins[2])[0]
            _, _, o_s, o_z = tensor(outs[0])
            params = (struct.pack("<fiI", o_s[0], int(o_z[0]), len(w_s))
                      + w_s.astype("<f4").tobytes())
            if b == B.CONV_2D:
                o = options(op, tflite.Conv2DOptions)
                head = struct.pack("<BBBBB", OP_CONV, ACT[o.FusedActivationFunction()],
                                   PAD[o.Padding()], o.StrideH(), o.StrideW())
            else:
                o = options(op, tflite.FullyConnectedOptions)
                head = struct.pack("<BB", OP_DENSE, ACT[o.FusedActivationFunction()])
                w = np.ascontiguousarray(w.T)  # [out, in] -> [in, out]
            layers.append(head + params + tensor_record(w) + tensor_record(bias))
        elif b == B.MAX_POOL_2D:
            o = options(op, tflite.Pool2DOptions)
            layers.append(struct.pack("<BBBBBB", OP_POOL, o.FilterHeight(), o.FilterWidth(),
                                      o.StrideH(), o.StrideW(), PAD[o.Padding()]))
        elif b in (B.RESHAPE, B.SOFTMAX):
            continue
        else:
            raise NotImplementedError("operator %d" % b)
    h, w, c = in_shape[1:]
    return (b"PTM1" + struct.pack("<BHHHfi", 1, h, w, c, in_s[0], int(in_z[0]))
            + struct.pack("<I", len(layers)) + b"".join(layers))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    blob = convert(os.path.join(HERE, "..", "mnist", "mnist_int8.tflite"))
    with open(os.path.join(out, "mnist_int8.ptm"), "wb") as f:
        f.write(blob)
    x, y = mnist_data.load("test")
    with open(os.path.join(out, "mnist_test.bin"), "wb") as f:
        f.write(struct.pack("<IHHH", len(x), 28, 28, 1))
        f.write(np.ascontiguousarray(x, np.uint8).tobytes())
        f.write(np.ascontiguousarray(y, np.uint8).tobytes())
    print("wrote %s/mnist_int8.ptm (%d bytes) and %s/mnist_test.bin"
          % (out, len(blob), out))


if __name__ == "__main__":
    main()
