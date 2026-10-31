"""Write a small sequential CNN/MLP as a .ptm file for C++ runners.

The .ptm format holds each layer's weights in the storage mode chosen for
it (float32, any PackedRomTensor format, Hough codes, or int8 for the
full-integer path), so a device-side runner can build RomTensor /
PackedRomTensor / HoughRomTensor objects straight from the file.
examples/mnist_modes/mnist_modes.cpp is the reference reader.

Layout (little endian):

    "PTM1"  u8 kind (0 float activations, 1 int8 activations)
    u16 in_h, in_w, in_c   f32 in_scale  i32 in_zp   (int8 input quantisation)
    u32 n_layers, then per layer:
      u8 op: 1 conv2d, 2 maxpool2d, 3 dense
      conv2d:  u8 act, u8 padding, u8 stride_h, u8 stride_w, filter (OHWI), bias
      maxpool: u8 kh, u8 kw, u8 stride_h, u8 stride_w, u8 padding
      dense:   u8 act, weights [in, out], bias
      int8 conv2d/dense additionally: f32 out_scale, i32 out_zp,
               u32 n_scales, f32 scales[n_scales]   (before the tensors)
    tensor:
      u8 storage: 0 f32, 1 packed, 2 hough, 3 int8, 4 int32
      u8 rank, u16 dims[rank]
      packed: u8 format, u8 bits, f32 scale, i32 zero_point
      hough:  u8 line_bits, u8 t_bits, u16 pair_stride, f32 lines[4 * 2^line_bits]
      u32 nbytes, data[nbytes]

Activations: 0 none, 1 relu, 3 relu6. Padding: 1 VALID, 2 SAME.
"""
import struct

import numpy as np

from .qat import Packed, RawF32, HoughEncoded

OP_CONV, OP_POOL, OP_DENSE = 1, 2, 3
ST_F32, ST_PACKED, ST_HOUGH, ST_I8, ST_I32 = 0, 1, 2, 3, 4
ACT = {"linear": 0, "relu": 1, "relu6": 3}
PAD = {"valid": 1, "same": 2}


def _dims(shape):
    return struct.pack("<B", len(shape)) + b"".join(struct.pack("<H", int(d)) for d in shape)


def tensor_record(t):
    """Serialise an Encoded tensor, or a raw int8/int32/float32 NumPy array."""
    if isinstance(t, np.ndarray):
        st = {np.dtype(np.int8): ST_I8, np.dtype(np.int32): ST_I32,
              np.dtype(np.float32): ST_F32}[t.dtype]
        data = np.ascontiguousarray(t).astype(t.dtype.newbyteorder("<")).tobytes()
        return struct.pack("<B", st) + _dims(t.shape) + struct.pack("<I", len(data)) + data
    if isinstance(t, RawF32):
        return tensor_record(t.values)
    if isinstance(t, Packed):
        data = t.payload.tobytes()
        return (struct.pack("<B", ST_PACKED) + _dims(t.shape)
                + struct.pack("<BBfi", t.fmt, t.bits, t.scale, t.zero_point)
                + struct.pack("<I", len(data)) + data)
    if isinstance(t, HoughEncoded):
        c = t.code
        data = c.stream().tobytes()
        return (struct.pack("<B", ST_HOUGH) + _dims(t.shape)
                + struct.pack("<BBH", c.line_bits, c.t_bits, c.stride)
                + c.lines.astype("<f4").tobytes()
                + struct.pack("<I", len(data)) + data)
    raise TypeError(type(t))


def _layer_act(layer):
    name = getattr(layer, "activation", None)
    name = getattr(name, "__name__", "linear") if name is not None else "linear"
    if name not in ACT:
        raise NotImplementedError("activation %s" % name)
    return ACT[name]


def _header(kind, input_shape, in_scale=1.0, in_zp=0):
    h, w, c = input_shape
    return b"PTM1" + struct.pack("<BHHHfi", kind, h, w, c, in_scale, in_zp)


def write_float_model(path, model, encoded):
    """Write a Keras Sequential model whose Conv2D/Dense kernels were encoded
    by QATModel.export() (float activations)."""
    enc = {id(layer): (e, b) for layer, e, b in encoded}
    layers = []
    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "Conv2D":
            e, b = enc[id(layer)]
            sh, sw = layer.strides
            layers.append(struct.pack("<BBBBB", OP_CONV, _layer_act(layer), PAD[layer.padding], sh, sw)
                          + tensor_record(e) + tensor_record(b))
        elif cls == "Dense":
            e, b = enc[id(layer)]
            layers.append(struct.pack("<BB", OP_DENSE, _layer_act(layer))
                          + tensor_record(e) + tensor_record(b))
        elif cls == "MaxPooling2D":
            kh, kw = layer.pool_size
            sh, sw = layer.strides
            layers.append(struct.pack("<BBBBBB", OP_POOL, kh, kw, sh, sw, PAD[layer.padding]))
        elif cls in ("Flatten", "Softmax", "InputLayer", "Dropout"):
            continue  # NHWC flatten is a no-op; the runner takes argmax
        else:
            raise NotImplementedError(cls)
    blob = _header(0, model.input_shape[1:]) + struct.pack("<I", len(layers)) + b"".join(layers)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


# ---------------------------------------------------------------------------
# Full-integer int8 (TFLite-style post-training quantisation)
# ---------------------------------------------------------------------------

def _act_qparams(lo, hi):
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = (hi - lo) / 255.0 or 1e-8
    zp = int(np.clip(np.rint(-128 - lo / scale), -128, 127))
    return np.float32(scale), zp


def write_int8_model(path, model, calib_x, input_scale=1.0 / 255.0, input_zp=-128):
    """Post-training full-integer quantisation (int8 activations, per-channel
    symmetric int8 weights, int32 biases), as TFLite does it, and write it.

    calib_x: float inputs used to find each layer's activation range.
    Returns (file bytes, weight bytes)."""
    import tensorflow as tf
    probes = [l for l in model.layers if type(l).__name__ in ("Conv2D", "Dense")]
    sub = tf.keras.Model(model.inputs, [l.output for l in probes])
    outs = sub.predict(calib_x, batch_size=256, verbose=0)
    ranges = {id(l): (o.min(), o.max()) for l, o in zip(probes, outs)}

    in_scale, in_zp = np.float32(input_scale), int(input_zp)
    layers, weight_bytes = [], 0
    for layer in model.layers:
        cls = type(layer).__name__
        if cls in ("Conv2D", "Dense"):
            k, b = layer.kernel.numpy(), layer.bias.numpy()
            if cls == "Conv2D":
                k = np.transpose(k, (3, 0, 1, 2))  # OHWI
                per_out = np.abs(k).reshape(k.shape[0], -1).max(1)
                w_scales = (per_out / 127.0).astype(np.float32)
                w_scales[w_scales == 0] = 1e-8
                q = np.clip(np.rint(k / w_scales[:, None, None, None]), -127, 127).astype(np.int8)
            else:  # [in, out], per output channel
                per_out = np.abs(k).max(0)
                w_scales = (per_out / 127.0).astype(np.float32)
                w_scales[w_scales == 0] = 1e-8
                q = np.clip(np.rint(k / w_scales[None, :]), -127, 127).astype(np.int8)
            bias = np.rint(b / (in_scale * w_scales)).astype(np.int32)
            out_scale, out_zp = _act_qparams(*ranges[id(layer)])
            params = struct.pack("<fiI", out_scale, out_zp, len(w_scales)) + w_scales.astype("<f4").tobytes()
            weight_bytes += q.size + bias.nbytes + 4 * len(w_scales) + 8
            if cls == "Conv2D":
                sh, sw = layer.strides
                head = struct.pack("<BBBBB", OP_CONV, _layer_act(layer), PAD[layer.padding], sh, sw)
            else:
                head = struct.pack("<BB", OP_DENSE, _layer_act(layer))
            layers.append(head + params + tensor_record(q) + tensor_record(bias))
            in_scale, in_zp = out_scale, out_zp
        elif cls == "MaxPooling2D":
            kh, kw = layer.pool_size
            sh, sw = layer.strides
            layers.append(struct.pack("<BBBBBB", OP_POOL, kh, kw, sh, sw, PAD[layer.padding]))
        elif cls in ("Flatten", "Softmax", "InputLayer", "Dropout"):
            continue
        else:
            raise NotImplementedError(cls)
    blob = (_header(1, model.input_shape[1:], input_scale, input_zp)
            + struct.pack("<I", len(layers)) + b"".join(layers))
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob), weight_bytes
