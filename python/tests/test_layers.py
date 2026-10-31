"""NumPy reference checks for the extended layer types."""
import math
import unittest

import numpy as np

import ptensor

RNG = np.random.default_rng(1234)


def rand(*shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=shape).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BinaryTest(unittest.TestCase):
    CASES = [((2, 3, 4), (2, 3, 4)), ((2, 3, 4), (4,)), ((2, 1, 4), (3, 1)),
             ((1, 5), (4, 1)), ((2, 2, 3, 4), (2, 1, 1, 4)), ((3,), (1,))]

    def test_ops_match_numpy(self):
        refs = {"sub": np.subtract, "mul": np.multiply, "maximum": np.maximum,
                "minimum": np.minimum, "add": np.add,
                "squared_difference": lambda a, b: (a - b) ** 2}
        for name, ref in refs.items():
            for sa, sb in self.CASES:
                a, b = rand(*sa), rand(*sb)
                got = ptensor.binary(name, a, b)
                np.testing.assert_allclose(got, ref(a, b), rtol=1e-6, atol=1e-6,
                                           err_msg="%s %s %s" % (name, sa, sb))

    def test_div_and_pow(self):
        a, b = rand(2, 3, lo=0.5, hi=3), rand(3, lo=0.5, hi=3)
        np.testing.assert_allclose(ptensor.div(a, b), a / b, rtol=1e-6)
        np.testing.assert_allclose(ptensor.power(a, b), a ** b, rtol=1e-5)

    def test_incompatible_shapes_rejected(self):
        with self.assertRaises(ValueError):
            ptensor.mul(rand(2, 3), rand(4))


class UnaryTest(unittest.TestCase):
    def test_activations(self):
        x = rand(64, lo=-6, hi=6)
        erf_free_gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        a, s = 1.6732632423543772, 1.0507009873554805
        hs = np.clip(x / 6 + 0.5, 0, 1)
        refs = {
            "selu": s * np.where(x >= 0, x, a * (np.exp(x) - 1)),
            "hard_sigmoid": hs,
            "hard_swish": x * hs,
            "swish": x * sigmoid(x),
            "gelu": erf_free_gelu,
            "softplus": np.log1p(np.exp(x)),
            "softsign": x / (1 + np.abs(x)),
            "abs": np.abs(x), "neg": -x, "square": x * x, "exp": np.exp(x),
            "floor": np.floor(x), "ceil": np.ceil(x), "round": np.round(x),
            "sin": np.sin(x), "cos": np.cos(x),
        }
        for name, ref in refs.items():
            np.testing.assert_allclose(ptensor.unary(name, x), ref, rtol=1e-5,
                                       atol=1e-5, err_msg=name)

    def test_positive_domain_ops(self):
        x = rand(32, lo=0.1, hi=9)
        np.testing.assert_allclose(ptensor.unary("sqrt", x), np.sqrt(x), rtol=1e-6)
        np.testing.assert_allclose(ptensor.unary("rsqrt", x), 1 / np.sqrt(x), rtol=1e-6)
        np.testing.assert_allclose(ptensor.unary("log", x), np.log(x), rtol=1e-6, atol=1e-6)

    def test_parameterised(self):
        x = rand(4, 8, lo=-3, hi=3)
        np.testing.assert_allclose(ptensor.leaky_relu(x, 0.1), np.where(x >= 0, x, 0.1 * x), rtol=1e-6)
        np.testing.assert_allclose(ptensor.elu(x, 0.5), np.where(x >= 0, x, 0.5 * (np.exp(x) - 1)),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(ptensor.clip(x, -1, 0.5), np.clip(x, -1, 0.5))
        self.assertEqual(ptensor.unary("abs", x).shape, x.shape)
        with self.assertRaises(ptensor.PTensorError):
            ptensor.clip(x, 1, -1)

    def test_every_op_is_listed(self):
        self.assertEqual(len(ptensor.UNARY_OPS), 22)


class LayerTest(unittest.TestCase):
    def test_batch_norm(self):
        x = rand(2, 3, 3, 5)
        mean, var = rand(5), rand(5, lo=0.1, hi=2)
        gamma, beta = rand(5), rand(5)
        ref = (x - mean) / np.sqrt(var + 1e-3) * gamma + beta
        np.testing.assert_allclose(ptensor.batch_norm(x, mean, var, gamma, beta),
                                   ref, rtol=1e-5, atol=1e-5)

    def test_pad2d(self):
        x = rand(1, 3, 4, 2)
        got = ptensor.pad2d(x, 1, 2, 0, 3, value=-7)
        ref = np.pad(x, ((0, 0), (1, 2), (0, 3), (0, 0)), constant_values=-7)
        np.testing.assert_array_equal(got, ref)

    def test_concat(self):
        for sa, sb, axis in [((2, 3), (2, 4), 1), ((2, 3), (5, 3), 0),
                             ((1, 2, 2, 3), (1, 2, 2, 1), -1),
                             ((1, 2, 2, 3), (1, 1, 2, 3), 1), ((3,), (2,), 0)]:
            a, b = rand(*sa), rand(*sb)
            np.testing.assert_array_equal(ptensor.concat(a, b, axis),
                                          np.concatenate([a, b], axis))

    def test_global_avgpool(self):
        x = rand(2, 4, 5, 3)
        np.testing.assert_allclose(ptensor.global_avgpool2d(x), x.mean(axis=(1, 2)),
                                   rtol=1e-5, atol=1e-6)


def depthwise_ref(x, f, b, stride, padding, mult, dilation):
    n, h, w, c = x.shape
    _, kh, kw, oc = f.shape
    sh, sw = stride
    dh, dw = dilation
    ekh, ekw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    if padding == "SAME":
        oh, ow = math.ceil(h / sh), math.ceil(w / sw)
        ph = max((oh - 1) * sh + ekh - h, 0) // 2
        pw = max((ow - 1) * sw + ekw - w, 0) // 2
    else:
        oh, ow = (h - ekh) // sh + 1, (w - ekw) // sw + 1
        ph = pw = 0
    out = np.zeros((n, oh, ow, oc), np.float64)
    for bi in range(n):
        for oy in range(oh):
            for ox in range(ow):
                for ci in range(c):
                    for m in range(mult):
                        o = ci * mult + m
                        acc = 0.0 if b is None else float(b[o])
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * sh - ph + ky * dh
                                ix = ox * sw - pw + kx * dw
                                if 0 <= iy < h and 0 <= ix < w:
                                    acc += x[bi, iy, ix, ci] * f[0, ky, kx, o]
                        out[bi, oy, ox, o] = acc
    return out.astype(np.float32)


class DepthwiseTest(unittest.TestCase):
    def test_against_reference(self):
        for padding in ("SAME", "VALID"):
            for stride, mult, dil in [((1, 1), 1, (1, 1)), ((2, 2), 2, (1, 1)),
                                      ((1, 2), 1, (2, 2)), ((2, 1), 3, (1, 2))]:
                x = rand(1, 7, 6, 3)
                f = rand(1, 3, 3, 3 * mult)
                b = rand(3 * mult)
                got = ptensor.depthwise_conv2d(x, f, b, stride, padding, mult, dil)
                ref = depthwise_ref(x, f, b, stride, padding, mult, dil)
                np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5,
                                           err_msg="%s %s %s %s" % (padding, stride, mult, dil))

    def test_no_bias(self):
        x, f = rand(2, 4, 4, 2), rand(1, 2, 2, 2)
        np.testing.assert_allclose(ptensor.depthwise_conv2d(x, f, padding="VALID"),
                                   depthwise_ref(x, f, None, (1, 1), "VALID", 1, (1, 1)),
                                   rtol=1e-5, atol=1e-5)


class RecurrentTest(unittest.TestCase):
    def test_lstm_cell(self):
        B, I, H = 2, 3, 4
        x, h, c = rand(B, I), rand(B, H), rand(B, H)
        W, U, b = rand(I, 4 * H, lo=-.5, hi=.5), rand(H, 4 * H, lo=-.5, hi=.5), rand(4 * H)
        z = x @ W + h @ U + b
        i, f, g, o = (z[:, k * H:(k + 1) * H] for k in range(4))
        c_ref = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h_ref = sigmoid(o) * np.tanh(c_ref)
        h_new, c_new = ptensor.lstm_cell(x, h, c, W, U, b)
        np.testing.assert_allclose(c_new, c_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(h_new, h_ref, rtol=1e-5, atol=1e-5)

    def test_gru_cell(self):
        B, I, H = 3, 2, 5
        x, h = rand(B, I), rand(B, H)
        W, U = rand(I, 3 * H, lo=-.5, hi=.5), rand(H, 3 * H, lo=-.5, hi=.5)
        bx, bh = rand(3 * H), rand(3 * H)
        xg, hg = x @ W + bx, h @ U + bh
        sl = lambda a, k: a[:, k * H:(k + 1) * H]
        z = sigmoid(sl(xg, 0) + sl(hg, 0))
        r = sigmoid(sl(xg, 1) + sl(hg, 1))
        n = np.tanh(sl(xg, 2) + r * sl(hg, 2))
        ref = z * h + (1 - z) * n
        np.testing.assert_allclose(ptensor.gru_cell(x, h, W, U, bx, bh), ref,
                                   rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
