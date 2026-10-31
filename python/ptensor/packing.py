"""Pack model weights into pTensor's compressed storage formats.

The output of these helpers is what you embed in firmware for a
PackedRomTensor; decode_packed() shows exactly what operators will read.
"""
import ctypes
import enum

import numpy as np

from ._lib import lib
from .ops import _check


class PackedFormat(enum.IntEnum):
    UINT_BITS = 0
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8


_NUMPY_DTYPE = {
    PackedFormat.INT8: "<i1",
    PackedFormat.UINT8: "<u1",
    PackedFormat.INT16: "<i2",
    PackedFormat.UINT16: "<u2",
    PackedFormat.INT32: "<i4",
    PackedFormat.FLOAT16: "<f2",
    PackedFormat.FLOAT32: "<f4",
    PackedFormat.FLOAT64: "<f8",
}


def packed_size(fmt, count, bits=0):
    """Bytes needed to store `count` values."""
    return int(lib.ptensor_packed_size(int(fmt), bits, count))


def pack_bits(codes, bits):
    """Pack unsigned integer codes (each < 2**bits) into an LSB-first bit stream."""
    codes = np.ascontiguousarray(codes, dtype=np.uint32).reshape(-1)
    out = np.zeros(packed_size(PackedFormat.UINT_BITS, codes.size, bits), np.uint8)
    _check("pack_bits", lib.ptensor_pack_bits(
        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), codes.size, bits,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))
    return out


def encode(values, fmt):
    """Encode an array in a byte-aligned format (INT8 ... FLOAT64) as bytes."""
    fmt = PackedFormat(fmt)
    if fmt == PackedFormat.UINT_BITS:
        raise ValueError("use pack_bits for UINT_BITS")
    return np.ascontiguousarray(values, dtype=_NUMPY_DTYPE[fmt]).view(np.uint8).reshape(-1)


def decode_packed(data, fmt, count, bits=0, scale=1.0, zero_point=0):
    """Decode packed bytes to float32 exactly as a float PackedRomTensor reads them."""
    data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    need = packed_size(fmt, count, bits)
    if need == 0 or data.size < need:
        raise ValueError("buffer too small or invalid format")
    out = np.empty(count, np.float32)
    _check("decode_packed", lib.ptensor_decode_packed_f32(
        int(fmt), bits, scale, zero_point, data.ctypes.data_as(ctypes.c_void_p),
        count, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return out


def float_to_half(x):
    return int(lib.ptensor_float_to_half(float(x)))


def half_to_float(h):
    return float(lib.ptensor_half_to_float(int(h)))
