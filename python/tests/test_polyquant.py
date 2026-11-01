"""Polynomial (quadratic / cubic) quantisation: encoder and device decoder."""
import unittest

import numpy as np

from ptensor import polyquant, qat
from ptensor._lib import lib

RNG = np.random.default_rng(11)
HAS_POLY = bool(lib.ptensor_has_poly_quant())


def weights(n, kind="laplace"):
    if kind == "laplace":
        return (RNG.laplace(size=n) * 0.05).astype(np.float32)
    return (RNG.standard_normal(n) * 0.05).astype(np.float32)


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - b) ** 2))


class PolyEncoderTest(unittest.TestCase):
    def test_never_worse_than_linear(self):
        for kind in ("laplace", "gauss"):
            w = weights(4000, kind)
            for bits in (2, 3, 4, 6, 8):
                lin = qat.UniformFormat(bits).quantizer(w.shape).encode(w).decode()
                for degree in (2, 3):
                    code = polyquant.poly_quantize(w, degree, bits)
                    self.assertLessEqual(mse(code.decode(), w), mse(lin, w) * (1 + 1e-3),
                                         (kind, bits, degree))

    def test_cubic_beats_linear_on_heavy_tails(self):
        w = weights(20000, "laplace")
        lin = qat.UniformFormat(3).quantizer(w.shape).encode(w).decode()
        cub = polyquant.poly_quantize(w, 3, 3).decode()
        self.assertLess(mse(cub, w), 0.85 * mse(lin, w))

    def test_lut_is_the_polynomial(self):
        c = np.array([0.01, 0.2, -0.03, 0.1], np.float32)
        lut = polyquant.build_lut(c, 3, 4)
        t = np.linspace(-1, 1, 16)
        np.testing.assert_allclose(lut, c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3, atol=1e-6)

    def test_sizes(self):
        code = polyquant.poly_quantize(weights(1000), 2, 4)
        self.assertEqual(code.stream_bytes, 500)
        self.assertEqual(code.nbytes, 500 + 12 + 2 + 64)
        self.assertEqual(code.nbytes, polyquant.estimate_bytes(1000, 2, 4))
        code16 = polyquant.poly_quantize(weights(1000), 3, 16)
        self.assertFalse(code16.use_lut)
        self.assertEqual(code16.nbytes, 2000 + 16 + 2)

    def test_qat_modes_exist_for_every_integer_width(self):
        for bits in qat.POLY_BITS:
            for prefix in ("quad", "cubic"):
                fmt = qat.get_format("%s%d" % (prefix, bits))
                self.assertEqual(fmt.bits, bits)


@unittest.skipUnless(HAS_POLY, "built without PTENSOR_WITH_POLY_QUANT")
class PolyDeviceTest(unittest.TestCase):
    def test_device_decode_matches_numpy_bit_for_bit(self):
        for shape, degree, bits in [((40, 30), 2, 4), ((17,), 3, 3), ((5, 3, 3, 4), 3, 2),
                                    ((300, 7), 2, 8), ((128, 64), 3, 16), ((70000,), 2, 5)]:
            w = weights(int(np.prod(shape))).reshape(shape)
            code = polyquant.poly_quantize(w, degree, bits, iters=4)
            np.testing.assert_array_equal(code.decode(), code.decode_on_device())
            # Same values without the table (polynomial evaluated per read).
            code.use_lut = False
            np.testing.assert_array_equal(code.decode(), code.decode_on_device())

    def test_device_lut_builder_matches_numpy(self):
        import ctypes
        c = np.array([0.01, 0.2, -0.03, 0.1], np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        for degree in (1, 2, 3):
            for bits in (2, 7, 12):
                out = np.empty(1 << bits, np.float32)
                self.assertEqual(lib.ptensor_poly_build_lut(degree, bits, c.ctypes.data_as(fp),
                                                            out.ctypes.data_as(fp)), 0)
                np.testing.assert_array_equal(out, polyquant.build_lut(c, degree, bits))


if __name__ == "__main__":
    unittest.main()
