"""Check pTensor's kernels against NumPy reference implementations.

Run from the repository root after building the C API:
    cmake -S . -B build -DPTENSOR_BUILD_C_API=ON && cmake --build build
    PYTHONPATH=python python3 -m unittest discover -s python/tests -v
"""
import unittest

import numpy as np

import ptensor
from ptensor import PackedFormat

RNG = np.random.default_rng(1234)


def np_conv2d(x, f, b, stride, padding):
    n, h, w, c = x.shape
    oc, kh, kw, _ = f.shape
    sh, sw = stride
    if padding == "SAME":
        oh, ow = -(-h // sh), -(-w // sw)
        ph = max((oh - 1) * sh + kh - h, 0)
        pw = max((ow - 1) * sw + kw - w, 0)
        top, left = ph // 2, pw // 2
        x = np.pad(x, ((0, 0), (top, ph - top), (left, pw - left), (0, 0)))
    else:
        oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    out = np.zeros((n, oh, ow, oc), np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = x[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            out[:, i, j, :] = np.tensordot(patch, f, axes=([1, 2, 3], [1, 2, 3]))
    if b is not None:
        out += b
    return out.astype(np.float32)


def np_pool(x, k, s, fn):
    n, h, w, c = x.shape
    oh, ow = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    out = np.zeros((n, oh, ow, c), np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = fn(x[:, i * s[0]:i * s[0] + k[0], j * s[1]:j * s[1] + k[1], :], axis=(1, 2))
    return out


class FloatOps(unittest.TestCase):
    def test_version(self):
        self.assertTrue(ptensor.__version__)

    def test_matmul(self):
        a = RNG.standard_normal((7, 13)).astype(np.float32)
        b = RNG.standard_normal((13, 5)).astype(np.float32)
        np.testing.assert_allclose(ptensor.matmul(a, b), a @ b, rtol=1e-5, atol=1e-5)

    def test_fully_connected(self):
        x = RNG.standard_normal((3, 10)).astype(np.float32)
        w = RNG.standard_normal((10, 4)).astype(np.float32)
        b = RNG.standard_normal(4).astype(np.float32)
        np.testing.assert_allclose(ptensor.fully_connected(x, w, b), x @ w + b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ptensor.fully_connected(x, w, b, activation="relu"),
                                   np.maximum(x @ w + b, 0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ptensor.fully_connected(x, w), x @ w, rtol=1e-5, atol=1e-5)

    def test_add(self):
        a = RNG.standard_normal(50).astype(np.float32)
        b = RNG.standard_normal(50).astype(np.float32)
        np.testing.assert_allclose(ptensor.add(a, b), a + b, rtol=1e-6)

    def test_activations(self):
        x = np.linspace(-8, 8, 101, dtype=np.float32)
        np.testing.assert_allclose(ptensor.relu(x), np.maximum(x, 0))
        np.testing.assert_allclose(ptensor.relu6(x), np.clip(x, 0, 6))
        np.testing.assert_allclose(ptensor.sigmoid(x), 1 / (1 + np.exp(-x)), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ptensor.tanh(x), np.tanh(x), rtol=1e-5, atol=1e-6)

    def test_softmax(self):
        x = RNG.standard_normal((4, 9)).astype(np.float32) * 3
        e = np.exp(x - x.max(axis=1, keepdims=True))
        np.testing.assert_allclose(ptensor.softmax(x), e / e.sum(axis=1, keepdims=True), rtol=1e-5, atol=1e-6)

    def test_conv2d(self):
        for padding in ("SAME", "VALID"):
            for stride in ((1, 1), (2, 2), (1, 2)):
                x = RNG.standard_normal((1, 9, 8, 3)).astype(np.float32)
                f = RNG.standard_normal((4, 3, 3, 3)).astype(np.float32)
                b = RNG.standard_normal(4).astype(np.float32)
                with self.subTest(padding=padding, stride=stride):
                    got = ptensor.conv2d(x, f, b, stride=stride, padding=padding)
                    np.testing.assert_allclose(got, np_conv2d(x, f, b, stride, padding), rtol=1e-4, atol=1e-4)

    def test_conv2d_no_bias(self):
        x = RNG.standard_normal((2, 6, 6, 2)).astype(np.float32)
        f = RNG.standard_normal((3, 2, 2, 2)).astype(np.float32)
        np.testing.assert_allclose(ptensor.conv2d(x, f, padding="VALID"),
                                   np_conv2d(x, f, None, (1, 1), "VALID"), rtol=1e-4, atol=1e-4)

    def test_pooling(self):
        x = RNG.standard_normal((1, 8, 8, 3)).astype(np.float32)
        np.testing.assert_allclose(ptensor.maxpool2d(x, 2), np_pool(x, (2, 2), (2, 2), np.max))
        np.testing.assert_allclose(ptensor.avgpool2d(x, 2), np_pool(x, (2, 2), (2, 2), np.mean),
                                   rtol=1e-5, atol=1e-6)

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            ptensor.matmul(np.ones((2, 3)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            ptensor.conv2d(np.ones((1, 2, 2, 1)), np.ones((1, 3, 3, 1)), padding="VALID")


class QuantizedOps(unittest.TestCase):
    def test_quantize_dequantize(self):
        x = np.linspace(-1, 1, 41, dtype=np.float32)
        scale, zp = 1 / 127.0, 0
        q = ptensor.quantize(x, scale, zp)
        np.testing.assert_array_equal(q, np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8))
        np.testing.assert_allclose(ptensor.dequantize(q, scale, zp), (q.astype(np.float32) - zp) * scale)

    def test_fully_connected_int8_matches_float(self):
        x_f = RNG.uniform(-1, 1, (2, 16)).astype(np.float32)
        w_f = RNG.uniform(-0.5, 0.5, (16, 6)).astype(np.float32)
        b_f = RNG.uniform(-0.2, 0.2, 6).astype(np.float32)
        x_s, x_zp = 2 / 255.0, 0
        w_s = np.abs(w_f).max() / 127.0
        y_f = x_f @ w_f + b_f
        o_s = (y_f.max() - y_f.min()) / 255.0
        o_zp = int(np.round(-128 - y_f.min() / o_s))
        xq = np.clip(np.round(x_f / x_s) + x_zp, -128, 127).astype(np.int8)
        wq = np.clip(np.round(w_f / w_s), -127, 127).astype(np.int8)
        bq = np.round(b_f / (x_s * w_s)).astype(np.int32)
        yq = ptensor.fully_connected_int8(xq, x_s, x_zp, wq, w_s, bq, o_s, o_zp)
        # Exact integer reference of the TFLite kernel, up to rounding.
        acc = (xq.astype(np.int64) - x_zp) @ wq.astype(np.int64) + bq
        ref = np.clip(np.round(acc * (x_s * w_s / o_s)) + o_zp, -128, 127)
        self.assertLessEqual(np.abs(yq.astype(np.int64) - ref).max(), 1)
        y_back = (yq.astype(np.float32) - o_zp) * o_s
        np.testing.assert_allclose(y_back, y_f, atol=4 * o_s + 0.02)


def np_quantize_multiplier(m):
    """TFLite QuantizeMultiplier: m = q * 2^shift with q a Q31 int."""
    if m == 0:
        return 0, 0
    q, shift = np.frexp(m)
    q_fixed = int(round(q * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    return q_fixed, int(shift)


def np_requantize(acc, multiplier, shift):
    """TFLite MultiplyByQuantizedMultiplier, elementwise over an int array."""
    acc = np.asarray(acc, np.int64)
    left = max(shift, 0)
    right = max(-shift, 0)
    ab = (acc * (1 << left)) * multiplier
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    num = ab + nudge
    high = np.where(num >= 0, num // (1 << 31), -((-num) // (1 << 31)))
    mask = (1 << right) - 1
    remainder = high & mask
    threshold = (mask >> 1) + (high < 0)
    return (high >> right) + (remainder > threshold)


def np_conv2d_int8(xq, xs, xz, fq, fs, bq, ys, yz, stride, padding, relu):
    acc = np_conv2d((xq.astype(np.float64) - xz), fq.astype(np.float64),
                    None, stride, padding).astype(np.float64)
    acc = np.rint(acc).astype(np.int64)
    if bq is not None:
        acc += bq
    fs = np.broadcast_to(np.atleast_1d(fs), (fq.shape[0],))
    out = np.empty_like(acc)
    for c in range(fq.shape[0]):
        m, sh = np_quantize_multiplier(float(xs) * float(np.float32(fs[c])) / float(ys))
        out[..., c] = np_requantize(acc[..., c], m, sh)
    out += yz
    lo = max(-128, yz) if relu else -128
    return np.clip(out, lo, 127).astype(np.int8)


class QuantizedLayers(unittest.TestCase):
    """int8 kernels used by TFLite full-integer models, checked bit-exactly."""

    def _conv_case(self, n, h, w, c, oc, k, stride, padding, per_channel, relu):
        xq = RNG.integers(-128, 128, (n, h, w, c)).astype(np.int8)
        fq = RNG.integers(-127, 128, (oc, k, k, c)).astype(np.int8)
        bq = RNG.integers(-2000, 2000, oc).astype(np.int32)
        xs, xz = 0.02, -5
        fs = (RNG.uniform(0.002, 0.01, oc) if per_channel else 0.004)
        fs = np.asarray(fs, np.float32)
        ys, yz = 0.15, 3
        got = ptensor.conv2d_int8(xq, xs, xz, fq, fs, bq, ys, yz, stride, padding,
                                  "relu" if relu else None)
        want = np_conv2d_int8(xq, xs, xz, fq, fs, bq, ys, yz, (stride, stride),
                              padding, relu)
        np.testing.assert_array_equal(got, want)

    def test_conv2d_int8_matches_tflite_arithmetic(self):
        self._conv_case(2, 9, 7, 3, 5, 3, 1, "SAME", True, False)
        self._conv_case(1, 9, 7, 3, 5, 3, 2, "SAME", True, True)
        self._conv_case(1, 8, 8, 4, 6, 3, 1, "VALID", False, False)
        self._conv_case(3, 6, 5, 1, 4, 1, 1, "VALID", True, True)

    def test_conv2d_int8_without_bias(self):
        xq = RNG.integers(-128, 128, (1, 5, 5, 2)).astype(np.int8)
        fq = RNG.integers(-127, 128, (3, 3, 3, 2)).astype(np.int8)
        got = ptensor.conv2d_int8(xq, 0.02, 0, fq, 0.005, None, 0.1, 0)
        want = np_conv2d_int8(xq, 0.02, 0, fq, np.float32(0.005), None, 0.1, 0,
                              (1, 1), "SAME", False)
        np.testing.assert_array_equal(got, want)

    def test_maxpool2d_int8(self):
        x = RNG.integers(-128, 128, (2, 6, 8, 3)).astype(np.int8)
        got = ptensor.maxpool2d_int8(x, 2)
        want = x.reshape(2, 3, 2, 4, 2, 3).max(axis=(2, 4))
        np.testing.assert_array_equal(got, want)

    def test_fully_connected_int8_per_channel(self):
        xq = RNG.integers(-128, 128, (3, 20)).astype(np.int8)
        wq = RNG.integers(-127, 128, (20, 7)).astype(np.int8)
        bq = RNG.integers(-500, 500, 7).astype(np.int32)
        ws = RNG.uniform(0.002, 0.01, 7).astype(np.float32)
        xs, xz, ys, yz = 0.03, -10, 0.2, 4
        got = ptensor.fully_connected_int8(xq, xs, xz, wq, ws, bq, ys, yz)
        acc = (xq.astype(np.int64) - xz) @ wq.astype(np.int64) + bq
        want = np.empty_like(acc)
        for c in range(7):
            m, sh = np_quantize_multiplier(xs * float(ws[c]) / ys)
            want[:, c] = np_requantize(acc[:, c], m, sh)
        want = np.clip(want + yz, -128, 127).astype(np.int8)
        np.testing.assert_array_equal(got, want)

    def test_fully_connected_int8_more_outputs_than_inputs(self):
        # Used to be rejected by a bogus shape check in the kernel.
        xq = RNG.integers(-128, 128, (1, 4)).astype(np.int8)
        wq = RNG.integers(-127, 128, (4, 9)).astype(np.int8)
        y = ptensor.fully_connected_int8(xq, 0.05, 0, wq, 0.01, None, 0.1, 0)
        self.assertEqual(y.shape, (1, 9))

    def test_softmax_int8(self):
        logits = RNG.integers(-128, 128, (4, 10)).astype(np.int8)
        q = ptensor.softmax_int8(logits, 0.1, 0)
        probs = (q.astype(np.float32) + 128) / 256
        want = ptensor.softmax(0.1 * logits.astype(np.float32))
        np.testing.assert_allclose(probs, want, atol=1 / 256 + 1e-6)

    def test_large_elementwise_calls_are_chunked(self):
        # Tensor dims are uint16; flat calls over more values must still work.
        x = RNG.uniform(-3, 3, 200000).astype(np.float32)
        q = ptensor.quantize(x, 0.05, 1)
        v = x / np.float32(0.05)
        rounded = np.sign(v) * np.floor(np.abs(v) + 0.5)  # half away from zero
        np.testing.assert_array_equal(
            q, np.clip(rounded + 1, -128, 127).astype(np.int8))
        np.testing.assert_allclose(ptensor.dequantize(q, 0.05, 1),
                                   (q.astype(np.float32) - 1) * 0.05, rtol=1e-6)
        np.testing.assert_array_equal(ptensor.relu(x), np.maximum(x, 0))
        np.testing.assert_allclose(ptensor.tanh(x), np.tanh(x), atol=1e-6)


class Packing(unittest.TestCase):
    def test_bit_widths(self):
        for bits in range(1, 17):
            codes = RNG.integers(0, 2 ** bits, 77)
            packed = ptensor.pack_bits(codes, bits)
            self.assertEqual(packed.size, (77 * bits + 7) // 8)
            got = ptensor.decode_packed(packed, PackedFormat.UINT_BITS, 77, bits=bits)
            np.testing.assert_array_equal(got, codes.astype(np.float32))

    def test_six_bit_affine(self):
        codes = RNG.integers(0, 64, 100)
        packed = ptensor.pack_bits(codes, 6)
        got = ptensor.decode_packed(packed, PackedFormat.UINT_BITS, 100, bits=6, scale=0.05, zero_point=32)
        np.testing.assert_allclose(got, (codes - 32) * np.float32(0.05), rtol=1e-6)

    def test_seven_bit_density(self):
        self.assertEqual(ptensor.packed_size(PackedFormat.UINT_BITS, 800, bits=7), 700)

    def test_code_out_of_range_rejected(self):
        with self.assertRaises(ptensor.PTensorError):
            ptensor.pack_bits([64], 6)

    def test_byte_aligned_formats(self):
        from ptensor.packing import encode
        cases = {
            PackedFormat.INT8: np.array([-128, -1, 0, 127]),
            PackedFormat.UINT8: np.array([0, 1, 200, 255]),
            PackedFormat.INT16: np.array([-32768, -5, 0, 32767]),
            PackedFormat.UINT16: np.array([0, 7, 40000, 65535]),
            PackedFormat.INT32: np.array([-100000, 0, 5, 100000]),
            PackedFormat.FLOAT32: np.array([0.5, -1.25, 3e5, 1e-3], np.float32),
            PackedFormat.FLOAT64: np.array([0.5, -1.25, 3e5, 1e-3], np.float64),
        }
        for fmt, vals in cases.items():
            with self.subTest(fmt=fmt.name):
                got = ptensor.decode_packed(encode(vals, fmt), fmt, vals.size)
                np.testing.assert_allclose(got, vals.astype(np.float32), rtol=1e-6)

    def test_float16_matches_numpy(self):
        x = RNG.standard_normal(1000).astype(np.float32) * 100
        halfs = x.astype(np.float16)
        for v, h in zip(x[:200], halfs[:200]):
            self.assertEqual(ptensor.float_to_half(v), int(h.view(np.uint16)))
        from ptensor.packing import encode
        got = ptensor.decode_packed(encode(halfs, PackedFormat.FLOAT16), PackedFormat.FLOAT16, x.size)
        np.testing.assert_array_equal(got, halfs.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
