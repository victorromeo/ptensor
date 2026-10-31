"""Hough line-codebook encoder, storage formats, and their device decoders."""
import unittest

import numpy as np

import ptensor
from ptensor import hough, qat
from ptensor._lib import lib

RNG = np.random.default_rng(7)
HAS_HOUGH = bool(lib.ptensor_has_hough())


def correlated(n, rho=0.9):
    """n values whose neighbours (stride 1) are correlated."""
    x = RNG.standard_normal(n // 2)
    y = rho * x + np.sqrt(1 - rho ** 2) * RNG.standard_normal(n // 2)
    return np.stack([x, y], 1).ravel().astype(np.float32) * 0.1


class PairingTest(unittest.TestCase):
    def test_pairs_cover_every_value_once(self):
        for count in (1, 2, 3, 7, 8, 9, 64, 101):
            for stride in (1, 2, 3, 8):
                i0, i1 = hough.pair_indices(count, stride)
                self.assertEqual(len(i0), hough.pair_count(count, stride))
                used = np.concatenate([i0, i1[i1 >= 0]])
                self.assertEqual(sorted(used.tolist()), list(range(count)))

    def test_candidate_strides(self):
        self.assertEqual(hough.candidate_strides((16, 3, 3, 8), 1152), [1, 8, 24, 72])


@unittest.skipUnless(HAS_HOUGH, "built without PTENSOR_WITH_HOUGH")
class HoughDeviceTest(unittest.TestCase):
    def test_device_decode_matches_numpy_bit_for_bit(self):
        for shape, lb, tb in [((40, 30), 4, 4), ((17,), 2, 3), ((5, 3, 3, 4), 3, 1),
                              ((300, 7), 0, 6), ((128, 64), 6, 2)]:
            w = RNG.standard_normal(shape).astype(np.float32)
            code = hough.hough_quantize(w, lb, tb, iters=4)
            np.testing.assert_array_equal(code.decode(), code.decode_on_device())
            self.assertEqual(code.stream_bytes, int(lib.ptensor_hough_stream_size(
                lb, tb, code.stride, code.count)))

    def test_bad_parameters_are_rejected(self):
        self.assertEqual(lib.ptensor_hough_stream_size(9, 4, 1, 10), 0)
        self.assertEqual(lib.ptensor_hough_stream_size(8, 9, 1, 10), 0)
        self.assertEqual(lib.ptensor_hough_stream_size(2, 2, 0, 10), 0)


class HoughEncoderTest(unittest.TestCase):
    def test_finds_the_correlated_pairing_and_beats_scalar(self):
        w = correlated(4096).reshape(64, 64)
        code = hough.hough_quantize(w, 4, 4)
        self.assertEqual(code.stride, 1)
        mse_h = float(((code.decode() - w) ** 2).mean())
        # Best uniform scalar quantiser at the same 4 bits per value.
        uni = qat.UniformFormat(4).quantizer(w.shape).encode(w)
        mse_u = float(((uni.decode() - w) ** 2).mean())
        self.assertLess(mse_h, 0.5 * mse_u)

    def test_size_accounting(self):
        w = RNG.standard_normal((100, 10)).astype(np.float32)
        code = hough.hough_quantize(w, 3, 3, iters=3)
        self.assertEqual(code.nbytes, (500 * 6 + 7) // 8 + 8 * 16 + 4)
        self.assertEqual(code.nbytes, hough.estimate_bytes(1000, 3, 3))
        self.assertEqual(len(code.stream()), code.stream_bytes)

    def test_warm_start_does_not_get_worse(self):
        w = RNG.standard_normal((64, 32)).astype(np.float32)
        q = hough.HoughQuantizer(3, 3, iters=8)
        e0 = float(((q.encode(w).decode() - w) ** 2).sum())
        e1 = float(((q.encode(w, refit_iters=2).decode() - w) ** 2).sum())
        self.assertLessEqual(e1, e0 * (1 + 1e-6))


class FormatTest(unittest.TestCase):
    def test_uniform_formats_decode_like_packed_rom_tensor(self):
        w = (RNG.standard_normal(500) * 0.2).astype(np.float32)
        for fmt in [qat.UniformFormat(b) for b in (2, 3, 4, 6, 7)] + [
                qat.UniformFormat(8, "int8"), qat.UniformFormat(16, "int16")]:
            e = fmt.quantizer(w.shape).encode(w)
            dev = ptensor.decode_packed(e.payload, e.fmt, e.count, e.bits, e.scale, e.zero_point)
            np.testing.assert_array_equal(e.decode(), dev, err_msg=fmt.name)
            self.assertEqual(e.nbytes, fmt.estimate_bytes(w.shape))

    def test_float_formats(self):
        w = (RNG.standard_normal(64)).astype(np.float32)
        e16 = qat.FloatFormat("f16").quantizer(w.shape).encode(w)
        dev = ptensor.decode_packed(e16.payload, e16.fmt, 64)
        np.testing.assert_array_equal(e16.decode(), dev)
        e64 = qat.FloatFormat("f64").quantizer(w.shape).encode(w)
        np.testing.assert_array_equal(e64.decode(), w)
        self.assertEqual(e64.nbytes, 8 * 64)

    def test_hough_falls_back_on_small_tensors(self):
        f = qat.HoughFormat(4, 4)
        self.assertFalse(f.uses_hough((8, 3, 3, 1)))
        self.assertTrue(f.uses_hough((784, 32)))
        self.assertEqual(f.estimate_bytes((72,)), qat.UniformFormat(4).estimate_bytes((72,)))

    def test_every_mode_constructs(self):
        for name in qat.MODES:
            self.assertIsInstance(qat.get_format(name), qat.WeightFormat)


if __name__ == "__main__":
    unittest.main()
