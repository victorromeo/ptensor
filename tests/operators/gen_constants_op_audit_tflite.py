"""Regenerate constants_op_audit_tflite.hpp: int8 single-operator results from
TFLite, used by test_op_audit_tflite.cpp.

Needs TensorFlow (pip install tensorflow). Run from the repo root:
    python tests/operators/gen_constants_op_audit_tflite.py tests/operators/constants_op_audit_tflite.hpp

Each case is a one-layer Keras model converted to a fully int8 TFLite model
and run with the reference kernels (BUILTIN_REF). The header holds the input,
the output, the quantization of both, and the int8 weights / int32 bias with
their (per-channel) scales exactly as TFLite stores them.
"""
import sys

import numpy as np
import tensorflow as tf

RNG = np.random.default_rng(1234)


def convert(model, in_shape, lo=-3.0, hi=3.0):
    def rep():
        for _ in range(64):
            yield [RNG.uniform(lo, hi, (1,) + in_shape).astype(np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    it = tf.lite.Interpreter(
        model_content=c.convert(),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    return it


def arr(ctype, name, values):
    values = np.asarray(values).reshape(-1)
    if ctype == "float":
        body = ", ".join("%.9ef" % v for v in values)
    else:
        body = ", ".join(str(int(v)) for v in values)
    return "const %s %s[%d] = {%s};" % (ctype, name, len(values), body)


def qparams(detail):
    q = detail["quantization_parameters"]
    return np.asarray(q["scales"], np.float32), np.asarray(q["zero_points"], np.int32)


def run(name, layer, in_shape, expect_op, x=None, dilate=False):
    """dilate=True: the layer has a 5x5 kernel whose odd taps are zeroed, so
    it computes exactly a 3x3 kernel with dilation 2 (with the same SAME
    padding). The converter would otherwise rewrite a dilated conv into
    SPACE_TO_BATCH_ND + CONV_2D + BATCH_TO_SPACE_ND; zero weights quantize to
    0 and add nothing to the int32 sums, so the outputs are those of the
    dilated op. The emitted filter is the 3x3 one."""
    inp = tf.keras.Input(in_shape)
    model = tf.keras.Model(inp, layer(inp))
    if dilate:
        k = layer.get_weights()
        mask = np.zeros((5, 5, 1, 1), np.float32)
        mask[::2, ::2] = 1.0
        k[0] = k[0] * mask
        layer.set_weights(k)
    it = convert(model, in_shape)
    ops = [o["op_name"] for o in it._get_ops_details()]
    assert ops == [expect_op], (name, ops)
    op = it._get_ops_details()[0]
    d, od = it.get_input_details()[0], it.get_output_details()[0]
    if x is None:
        x = RNG.integers(-128, 128, (1,) + in_shape, dtype=np.int8)
    it.set_tensor(d["index"], x)
    it.invoke()
    y = it.get_tensor(od["index"])
    (si, zi), (so, zo) = d["quantization"], od["quantization"]
    lines = ["// %s: TFLite %s, input %s -> output %s" % (name, expect_op, list(x.shape), list(y.shape)),
             "const float %s_in_scale = %.9ef; const int32_t %s_in_zp = %d;" % (name, si, name, zi),
             "const float %s_out_scale = %.9ef; const int32_t %s_out_zp = %d;" % (name, so, name, zo),
             arr("int8_t", name + "_in", x), arr("int8_t", name + "_out", y)]
    details = {t["index"]: t for t in it.get_tensor_details()}
    if len(op["inputs"]) >= 2:  # weights and bias
        w = details[op["inputs"][1]]
        ws, _ = qparams(w)
        wv = it.get_tensor(w["index"])
        if dilate:  # [O or 1, 5, 5, I] -> [O or 1, 3, 3, I]
            assert not wv[:, 1::2, :, :].any() and not wv[:, :, 1::2, :].any()
            wv = wv[:, ::2, ::2, :]
        lines += ["// filter shape %s" % list(wv.shape),
                  arr("int8_t", name + "_w", wv),
                  arr("float", name + "_w_scale", ws)]
        if len(op["inputs"]) >= 3 and op["inputs"][2] >= 0:
            b = details[op["inputs"][2]]
            lines.append(arr("int32_t", name + "_b", it.get_tensor(b["index"])))
    return "\n".join(lines) + "\n"


def main(path):
    K = tf.keras.layers
    every = np.arange(-128, 128, dtype=np.int8).reshape(1, 256)
    cases = [
        run("conv_same", K.Conv2D(4, 3, padding="same"), (7, 6, 3), "CONV_2D"),
        run("conv_dil2", K.Conv2D(4, 5, padding="same"), (8, 7, 2), "CONV_2D", dilate=True),
        run("conv_1x1_s4", K.Conv2D(3, 1, strides=4, padding="same"), (8, 8, 2), "CONV_2D"),
        run("conv_2x2_s4_valid", K.Conv2D(3, 2, strides=4, padding="valid"), (11, 11, 2), "CONV_2D"),
        run("conv_2x2_s4_same", K.Conv2D(3, 2, strides=4, padding="same"), (8, 8, 2), "CONV_2D"),
        run("dw_dil2", K.DepthwiseConv2D(5, padding="same"), (8, 7, 3), "DEPTHWISE_CONV_2D",
            dilate=True),
        run("dw_s2_m2", K.DepthwiseConv2D(3, strides=2, padding="same", depth_multiplier=2),
            (7, 7, 2), "DEPTHWISE_CONV_2D"),
        run("dw_s12", K.DepthwiseConv2D(3, strides=(1, 2), padding="same"), (6, 9, 2),
            "DEPTHWISE_CONV_2D"),
        run("conv_s21", K.Conv2D(3, 3, strides=(2, 1), padding="same"), (7, 6, 2), "CONV_2D"),
        run("maxpool_same", K.MaxPool2D(3, 2, padding="same"), (7, 7, 2), "MAX_POOL_2D"),
        run("maxpool_valid", K.MaxPool2D(2, 2, padding="valid"), (6, 5, 3), "MAX_POOL_2D"),
        run("avgpool_same", K.AveragePooling2D(3, 2, padding="same"), (7, 7, 2), "AVERAGE_POOL_2D"),
        run("avgpool_valid", K.AveragePooling2D(2, 2, padding="valid"), (6, 5, 3), "AVERAGE_POOL_2D"),
        run("logistic", K.Activation("sigmoid"), (256,), "LOGISTIC", every),
        run("tanh", K.Activation("tanh"), (256,), "TANH", every),
        run("softmax_2d", K.Softmax(), (10,), "SOFTMAX"),
        run("softmax_4d", K.Softmax(), (2, 3, 6), "SOFTMAX"),
    ]
    text = "\n".join([
        "// Generated by tests/operators/gen_constants_op_audit_tflite.py with TensorFlow %s." % tf.__version__,
        "#ifndef CONSTANTS_OP_AUDIT_TFLITE_HPP\n#define CONSTANTS_OP_AUDIT_TFLITE_HPP\n#include <stdint.h>\n",
        "\n".join(cases),
        "#endif\n"])
    with open(path, "w") as f:  # the converter prints to stdout
        f.write(text)


if __name__ == "__main__":
    main(sys.argv[1])
