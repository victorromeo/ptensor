"""Structured pruning: channel selection, and pruned networks running on pTensor.

The NumPy tests always run; the Keras tests run when TensorFlow is installed.
"""
import unittest

import numpy as np

import ptensor
from ptensor import prune

RNG = np.random.default_rng(5)

try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False


def relu(x):
    return np.maximum(x, 0)


class SelectionTest(unittest.TestCase):
    def test_scores(self):
        k = np.array([[1.0, -2.0, 0.0], [-3.0, 0.5, 0.0]])
        np.testing.assert_allclose(prune.filter_scores(k, "l1"), [4.0, 2.5, 0.0])
        np.testing.assert_allclose(prune.filter_scores(k, "l2"), [np.sqrt(10), np.sqrt(4.25), 0.0])
        conv = RNG.standard_normal((3, 3, 2, 5))  # HWIO
        np.testing.assert_allclose(prune.filter_scores(conv), np.abs(conv).sum((0, 1, 2)))
        with self.assertRaises(ValueError):
            prune.filter_scores(k, "l0")

    def test_n_keep(self):
        self.assertEqual(prune.n_keep(16, 0.5), 8)
        self.assertEqual(prune.n_keep(10, 0.75), 2)   # 2.5 rounds to even
        self.assertEqual(prune.n_keep(8, 0.9), 1)     # never zero
        self.assertEqual(prune.n_keep(32, 0.3, multiple_of=4), 24)
        self.assertEqual(prune.n_keep(5, 0.0), 5)
        with self.assertRaises(ValueError):
            prune.n_keep(4, 1.0)

    def test_keep_indices_sorted_and_deterministic(self):
        s = np.array([1.0, 5.0, 3.0, 5.0, 0.5])
        np.testing.assert_array_equal(prune.keep_indices(s, 3), [1, 2, 3])
        np.testing.assert_array_equal(prune.keep_indices(s, 1), [1])  # tie -> lower index

    def test_zero_units_removed_exactly(self):
        # Units whose incoming weights are all zero output relu(0 + bias); with
        # zero bias they contribute nothing, so removing them must not change
        # the network at all.
        w1 = RNG.standard_normal((6, 8)).astype(np.float32)
        b1 = RNG.standard_normal(8).astype(np.float32)
        dead = [1, 4, 6, 7]
        w1[:, dead] = 0
        b1[dead] = 0
        w2 = RNG.standard_normal((8, 3)).astype(np.float32)
        b2 = RNG.standard_normal(3).astype(np.float32)
        (pw1, pb1), (pw2, pb2) = prune.prune_dense_chain([(w1, b1), (w2, b2)], 0.5)[0]
        self.assertEqual(pw1.shape, (6, 4))
        self.assertEqual(pw2.shape, (4, 3))
        x = RNG.standard_normal((5, 6)).astype(np.float32)
        full = relu(x @ w1 + b1) @ w2 + b2
        small = relu(x @ pw1 + pb1) @ pw2 + pb2
        np.testing.assert_allclose(small, full, rtol=1e-5, atol=1e-5)

    def test_pruned_mlp_on_ptensor(self):
        # The pruned network is a plain smaller MLP: pTensor's fully
        # connected kernel runs it and matches NumPy.
        ws = [(RNG.standard_normal((20, 16)).astype(np.float32), RNG.standard_normal(16).astype(np.float32)),
              (RNG.standard_normal((16, 12)).astype(np.float32), RNG.standard_normal(12).astype(np.float32)),
              (RNG.standard_normal((12, 4)).astype(np.float32), RNG.standard_normal(4).astype(np.float32))]
        small, kept = prune.prune_dense_chain(ws, 0.5)
        self.assertEqual([len(k) for k in kept], [8, 6])
        x = RNG.standard_normal((3, 20)).astype(np.float32)
        ref, out = x, x
        for i, (k, b) in enumerate(small):
            last = i == len(small) - 1
            ref = ref @ k + b if last else relu(ref @ k + b)
            out = ptensor.fully_connected(out, k, b, activation=None if last else "relu")
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@unittest.skipUnless(HAS_TF, "needs TensorFlow")
class KerasPruneTest(unittest.TestCase):
    def model(self, bn=False):
        L = tf.keras.layers
        tf.keras.utils.set_random_seed(1)
        layers = [L.Input((8, 8, 2)), L.Conv2D(6, 3, padding="same", name="c1")]
        if bn:
            layers.append(L.BatchNormalization(name="bn"))
        layers += [L.ReLU(), L.MaxPooling2D(2), L.Conv2D(8, 3, padding="same", activation="relu", name="c2"),
                   L.MaxPooling2D(2), L.Flatten(), L.Dense(10, activation="relu", name="d1"),
                   L.Dense(3, name="out")]
        m = tf.keras.Sequential(layers)
        if bn:
            bnl = m.get_layer("bn")
            bnl.set_weights([RNG.uniform(0.5, 2, 6).astype(np.float32) for _ in range(2)]
                            + [RNG.standard_normal(6).astype(np.float32), RNG.uniform(0.5, 2, 6).astype(np.float32)])
        return m

    def zero_and_compare(self, m, chosen):
        # Zero the dropped channels in the original (output weights and
        # biases, and BN beta so the dropped channels output exactly 0
        # after the ReLU); the pruned model must then match it exactly.
        x = RNG.standard_normal((4, 8, 8, 2)).astype(np.float32)
        for name, keep in chosen.items():
            layer = m.get_layer(name)
            drop = np.setdiff1d(np.arange(layer.kernel.shape[-1]), keep)
            k, b = layer.get_weights()
            k[..., drop] = 0
            b[drop] = 0
            layer.set_weights([k, b])
            if name == "c1" and any(l.name == "bn" for l in m.layers):
                g, beta, mean, var = m.get_layer("bn").get_weights()
                beta[drop] = -1e3  # BN output strongly negative, ReLU -> 0
                m.get_layer("bn").set_weights([g, beta, mean, var])
        small = prune.apply_pruning(m, chosen)
        np.testing.assert_allclose(small.predict(x, verbose=0), m.predict(x, verbose=0), rtol=1e-4, atol=1e-4)
        return small

    def test_shapes_and_names(self):
        m = self.model()
        self.assertEqual(prune.prunable_layers(m), ["c1", "c2", "d1"])
        small = prune.prune(m, 0.5)
        self.assertEqual([l.name for l in small.layers if hasattr(l, "kernel")], ["c1", "c2", "d1", "out"])
        self.assertEqual(small.get_layer("c1").kernel.shape, (3, 3, 2, 3))
        self.assertEqual(small.get_layer("c2").kernel.shape, (3, 3, 3, 4))
        self.assertEqual(small.get_layer("d1").kernel.shape, (2 * 2 * 4, 5))
        self.assertEqual(small.get_layer("out").kernel.shape, (5, 3))
        n0, s0, m0 = prune.model_stats(m)
        n1, s1, m1 = prune.model_stats(small)
        self.assertLess(s1, s0 / 2)
        self.assertLess(m1, m0 / 2)

    def test_exact_through_flatten(self):
        m = self.model()
        chosen = {"c1": np.array([0, 2, 5]), "c2": np.array([1, 3, 4, 7]), "d1": np.arange(0, 10, 2)}
        self.zero_and_compare(m, chosen)

    def test_exact_with_batchnorm(self):
        m = self.model(bn=True)
        chosen = {"c1": np.array([1, 2, 4, 5])}
        small = self.zero_and_compare(m, chosen)
        self.assertEqual(small.get_layer("bn").get_weights()[0].shape, (4,))

    def test_per_layer_ratio_and_stepped_fit(self):
        m = self.model()
        x = RNG.standard_normal((64, 8, 8, 2)).astype(np.float32)
        y = RNG.integers(0, 3, 64)
        p = prune.PrunedModel(m, {"c2": 0.5, "d1": 0.8}, log=lambda *a: None)
        p.fit(x, y, epochs=1, steps=2, batch_size=32)
        self.assertEqual(p.model.get_layer("c1").kernel.shape[-1], 6)
        self.assertEqual(p.model.get_layer("c2").kernel.shape[-1], 4)
        self.assertEqual(p.model.get_layer("d1").kernel.shape[-1], 2)
        # kept indices refer to the original model's channels
        self.assertEqual(len(p.kept["c2"]), 4)
        self.assertTrue(np.all(p.kept["c2"] < 8))
        self.assertEqual(m.get_layer("c2").kernel.shape[-1], 8)  # original untouched
        with self.assertRaises(ValueError):
            prune.PrunedModel(m, {"out": 0.5})


if __name__ == "__main__":
    unittest.main()
