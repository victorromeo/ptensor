"""Locate and load the pTensor C API shared library (libptensor_c)."""
import ctypes
import os
import sys
from pathlib import Path

_LIB_NAMES = {
    "linux": "libptensor_c.so",
    "darwin": "libptensor_c.dylib",
    "win32": "ptensor_c.dll",
}


def _candidates():
    env = os.environ.get("PTENSOR_LIBRARY")
    if env:
        yield Path(env)
    name = _LIB_NAMES.get(sys.platform, "libptensor_c.so")
    here = Path(__file__).resolve().parent
    yield here / name
    # Fall back to CMake build trees next to the repository checkout.
    repo = here.parent.parent
    for build in sorted(repo.glob("build*")):
        yield build / "src" / "pTensor" / name


def load():
    tried = []
    for path in _candidates():
        tried.append(str(path))
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise OSError(
        "Could not find the pTensor C library. Build it with\n"
        "  cmake -S . -B build -DPTENSOR_BUILD_C_API=ON && cmake --build build\n"
        "or point PTENSOR_LIBRARY at libptensor_c. Looked in:\n  "
        + "\n  ".join(tried)
    )


lib = load()

_f32p = ctypes.POINTER(ctypes.c_float)
_i8p = ctypes.POINTER(ctypes.c_int8)
_i32p = ctypes.POINTER(ctypes.c_int32)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_u8p = ctypes.POINTER(ctypes.c_uint8)
_u32 = ctypes.c_uint32
_i32 = ctypes.c_int32
_int = ctypes.c_int
_f32 = ctypes.c_float


def _sig(name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_sig("ptensor_version", ctypes.c_char_p)
_sig("ptensor_last_error", _u32)
_sig("ptensor_conv_output_size", _u32, _u32, _u32, _u32, _int)
_sig("ptensor_matmul_f32", _int, _f32p, _f32p, _f32p, _u32, _u32, _u32)
_sig("ptensor_fully_connected_f32", _int, _f32p, _f32p, _f32p, _f32p, _u32, _u32, _u32, _int)
_sig("ptensor_add_f32", _int, _f32p, _f32p, _f32p, _u32)
for _n in ("relu", "relu6", "sigmoid", "tanh"):
    _sig("ptensor_%s_f32" % _n, _int, _f32p, _f32p, _u32)
_sig("ptensor_softmax_f32", _int, _f32p, _f32p, _u32, _u32)
_sig("ptensor_conv2d_f32", _int, _f32p, _u32, _u32, _u32, _u32, _f32p, _u32,
     _u32, _u32, _f32p, _u32, _u32, _int, _f32p)
for _n in ("maxpool2d", "avgpool2d"):
    _sig("ptensor_%s_f32" % _n, _int, _f32p, _u32, _u32, _u32, _u32, _u32,
         _u32, _u32, _u32, _int, _f32p)
_sig("ptensor_quantize_i8", _int, _f32p, _i8p, _u32, _f32, _i32)
_sig("ptensor_dequantize_i8", _int, _i8p, _f32p, _u32, _f32, _i32)
_sig("ptensor_fully_connected_i8", _int, _i8p, _f32, _i32, _i8p, _f32, _i32p,
     _i8p, _f32, _i32, _u32, _u32, _u32, _int)
_sig("ptensor_fully_connected_i8_pc", _int, _i8p, _f32, _i32, _i8p, _f32p, _i32p,
     _i8p, _f32, _i32, _u32, _u32, _u32, _int)
_sig("ptensor_conv2d_i8", _int, _i8p, _u32, _u32, _u32, _u32, _f32, _i32, _i8p,
     _u32, _u32, _u32, _f32p, _u32, _i32p, _u32, _u32, _int, _int, _i8p, _f32,
     _i32)
_sig("ptensor_maxpool2d_i8", _int, _i8p, _u32, _u32, _u32, _u32, _u32, _u32,
     _u32, _u32, _int, _i8p)
_sig("ptensor_packed_size", ctypes.c_size_t, _int, _u32, _u32)
_sig("ptensor_pack_bits", _int, _u32p, _u32, _u32, _u8p)
_sig("ptensor_decode_packed_f32", _int, _int, _u32, _f32, _i32, ctypes.c_void_p, _u32, _f32p)
_sig("ptensor_float_to_half", ctypes.c_uint16, _f32)
_sig("ptensor_has_hough", _int)
_sig("ptensor_hough_stream_size", ctypes.c_size_t, _u32, _u32, _u32, _u32)
_sig("ptensor_decode_hough_f32", _int, _u32, _u32, _u32, _f32p, ctypes.c_void_p, _u32, _f32p)
_sig("ptensor_half_to_float", _f32, ctypes.c_uint16)
_sig("ptensor_binary_f32", _int, _int, _f32p, _u32p, _u32, _f32p, _u32p, _u32, _f32p)
_sig("ptensor_unary_f32", _int, _int, _f32p, _f32p, _u32, _f32, _f32)
_sig("ptensor_batchnorm_f32", _int, _f32p, _u32, _u32, _f32p, _f32p, _f32p, _f32p, _f32, _f32p)
_sig("ptensor_pad2d_f32", _int, _f32p, _u32, _u32, _u32, _u32, _u32, _u32, _u32, _u32, _f32, _f32p)
_sig("ptensor_concat_f32", _int, _f32p, _u32p, _f32p, _u32p, _u32, _int, _f32p)
_sig("ptensor_global_avgpool_f32", _int, _f32p, _u32, _u32, _u32, _u32, _f32p)
_sig("ptensor_depthwise_conv2d_f32", _int, _f32p, _u32, _u32, _u32, _u32, _f32p,
     _u32, _u32, _u32, _f32p, _u32, _u32, _int, _u32, _u32, _f32p)
_sig("ptensor_lstm_cell_f32", _int, _f32p, _f32p, _f32p, _f32p, _f32p, _f32p,
     _u32, _u32, _u32, _f32p, _f32p)
_sig("ptensor_gru_cell_f32", _int, _f32p, _f32p, _f32p, _f32p, _f32p, _f32p,
     _u32, _u32, _u32, _f32p)
