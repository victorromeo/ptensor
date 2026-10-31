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
    conv_output_size,
    dequantize,
    fully_connected,
    fully_connected_int8,
    matmul,
    maxpool2d,
    quantize,
    relu,
    relu6,
    sigmoid,
    softmax,
    tanh,
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
]
