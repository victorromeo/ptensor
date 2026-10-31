"""Python bindings for pTensor.

These call the same C++ kernels that run on the microcontroller, so a model's
layers can be checked against NumPy (or TensorFlow) without writing any C.

    import numpy as np, ptensor
    y = ptensor.conv2d(x, w, b, stride=(1, 1), padding="SAME")
"""
from ._lib import lib as _lib
from .ops import (
    PTensorError,
    add,
    avgpool2d,
    conv2d,
    conv2d_int8,
    conv_output_size,
    dequantize,
    fully_connected,
    fully_connected_int8,
    matmul,
    maxpool2d,
    maxpool2d_int8,
    quantize,
    relu,
    relu6,
    sigmoid,
    softmax,
    softmax_int8,
    tanh,
    UNARY_OPS,
    batch_norm,
    binary,
    broadcast_add,
    clip,
    concat,
    depthwise_conv2d,
    div,
    elu,
    global_avgpool2d,
    gru_cell,
    leaky_relu,
    lstm_cell,
    maximum,
    minimum,
    mul,
    pad2d,
    power,
    squared_difference,
    sub,
    unary,
)
from .packing import (
    PackedFormat,
    decode_packed,
    float_to_half,
    half_to_float,
    pack_bits,
    packed_size,
)

__version__ = _lib.ptensor_version().decode()

__all__ = [
    "PTensorError", "add", "avgpool2d", "conv2d", "conv_output_size",
    "dequantize", "fully_connected", "fully_connected_int8", "matmul",
    "maxpool2d", "quantize", "relu", "relu6", "sigmoid", "softmax", "tanh",
    "PackedFormat", "decode_packed", "float_to_half", "half_to_float",
    "pack_bits", "packed_size",
    "UNARY_OPS",
    "conv2d_int8",
    "maxpool2d_int8",
    "softmax_int8",
    "batch_norm",
    "binary",
    "broadcast_add",
    "clip",
    "concat",
    "depthwise_conv2d",
    "div",
    "elu",
    "global_avgpool2d",
    "gru_cell",
    "leaky_relu",
    "lstm_cell",
    "maximum",
    "minimum",
    "mul",
    "pad2d",
    "power",
    "squared_difference",
    "sub",
    "unary",
]
