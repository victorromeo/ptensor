"""Fused operators and the fusion pass against the unfused operator chains.

The TFLite checks run only when TensorFlow is installed.
"""
import unittest

import numpy as np

import ptensor
from ptensor import fusion

RNG = np.random.default_rng(77)


def rand(*shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=shape).astype(np.float32)


def bn_params(c):
    return {"mean": rand(c, lo=-0.5, hi=0.5), "variance": rand(c, lo=0.1, hi=1.5),
            "gamma": rand(c, lo=-1.5, hi=1.5), "beta": rand(c, lo=-0.5, hi=0.5),
            "epsilon": 1e-3}


def batch_norm(y, bn):
    return ptensor.batch_norm(y, bn["mean"], bn["variance"], bn["gamma"], bn["beta"],
                              bn["epsilon"])


@unittest.skipUnless(fusion.available(), "built without PTENSOR_WITH_FUSION")
class FusedKernelTest(unittest.TestCase):
    def test_conv_activation_is_bit_identical(self):
        for padding, stride in (("SAME", 1), ("VALID", 1), ("SAME", 2)):
            x, w, b = rand(2, 9, 8, 3), rand(5, 3, 3, 3), rand(5)
            ref = np.clip(ptensor.conv2d(x, w, b, stride, padding), 0, 6)
            got = fusion.fused_conv2d(x, w, b, stride=stride, padding=padding,
                                      activation="relu6")
            np.testing.assert_array_equal(got, ref, err_msg="%s %s" % (padding, stride))

    def test_conv_batch_norm_fold(self):
        x, w, b = rand(1, 10, 10, 4), rand(6, 3, 3, 4), rand(6)
        bn = bn_params(6)
        ref = np.maximum(batch_norm(ptensor.conv2d(x, w, b), bn), 0)
        scale, bias = fusion.fold_batch_norm(bn["mean"], bn["variance"], bn["gamma"],
                                             bn["beta"], bn["epsilon"], bias=b)
        by_scale = fusion.fused_conv2d(x, w, bias, scale=scale, activation="relu")
        wf, bf = fusion.fold_conv2d(w, b, bn)
        by_weights = fusion.fused_conv2d(x, wf, bf, activation="relu")
        np.testing.assert_allclose(by_scale, ref, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(by_weights, ref, rtol=2e-5, atol=2e-5)

    def test_depthwise(self):
        x, w, b = rand(1, 8, 8, 4), rand(1, 3, 3, 8), rand(8)
        ref = np.clip(ptensor.depthwise_conv2d(x, w, b, depth_multiplier=2), -1, 1)
        got = fusion.fused_depthwise_conv2d(x, w, b, activation="relu1", depth_multiplier=2)
        np.testing.assert_array_equal(got, ref)
        bn = bn_params(8)
        ref = np.clip(batch_norm(ptensor.depthwise_conv2d(x, w, b, depth_multiplier=2), bn), 0, 6)
        wf, bf = fusion.fold_depthwise_conv2d(w, b, bn)
        got = fusion.fused_depthwise_conv2d(x, wf, bf, activation="relu6", depth_multiplier=2)
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)

    def test_dense(self):
        x, w, b = rand(4, 30), rand(30, 7), rand(7)
        ref = ptensor.fully_connected(x, w, b, activation="relu")
        np.testing.assert_array_equal(
            fusion.fused_fully_connected(x, w, b, activation="relu"), ref)
        bn = bn_params(7)
        ref = np.clip(batch_norm(ptensor.fully_connected(x, w, b), bn), -0.5, 0.5)
        wf, bf = fusion.fold_dense(w, b, bn)
        got = fusion.fused_fully_connected(x, wf, bf, activation=(-0.5, 0.5))
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)

    def test_add(self):
        a, b = rand(2, 4, 4, 6), rand(6)
        np.testing.assert_array_equal(fusion.fused_add(a, b, "relu"),
                                      np.maximum(ptensor.broadcast_add(a, b), 0))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            fusion.fused_conv2d(rand(1, 4, 4, 2), rand(3, 3, 3, 2), scale=rand(2))
        with self.assertRaises(ValueError):
            fusion.act_range((1.0, 0.0))
        with self.assertRaises(ValueError):
            fusion.act_range("swish")


@unittest.skipUnless(fusion.available(), "built without PTENSOR_WITH_FUSION")
class FusePassTest(unittest.TestCase):
    def network(self):
        return [
            {"op": "conv2d", "filters": rand(8, 3, 3, 3), "bias": rand(8), "stride": 2},
            {"op": "batch_norm", **bn_params(8)},
            {"op": "activation", "activation": "relu6"},
            {"op": "depthwise_conv2d", "filters": rand(1, 3, 3, 8), "bias": None},
            {"op": "batch_norm", **bn_params(8)},
            {"op": "activation", "activation": "relu"},
            {"op": "conv2d", "filters": rand(4, 1, 1, 8), "bias": rand(4)},
            {"op": "batch_norm", **bn_params(4)},
            {"op": "maxpool2d", "kernel": 2, "stride": 2},
            {"op": "flatten"},
            {"op": "dense", "weights": rand(4 * 4 * 4, 10), "bias": rand(10)},
            {"op": "activation", "activation": "relu"},
            {"op": "activation", "activation": "relu6"},
        ]

    def test_fused_network_matches_unfused(self):
        layers = self.network()
        x = rand(2, 16, 16, 3)
        ref = fusion.run(layers, x)
        for fold_weights in (True, False):
            fused, report = fusion.fuse(layers, fold_weights=fold_weights)
            self.assertEqual([l["op"] for l in fused],
                             ["conv2d", "depthwise_conv2d", "conv2d", "maxpool2d",
                              "flatten", "dense"])
            self.assertEqual(report[0], ("conv2d", ["batch_norm", "relu6"]))
            self.assertEqual(fused[-1]["activation"], (0.0, 6.0))
            np.testing.assert_allclose(fusion.run(fused, x), ref, rtol=1e-4, atol=1e-4)
        # The input list is left untouched.
        self.assertEqual(len(layers), 13)

    def test_batch_norm_after_activation_is_not_folded(self):
        layers = [{"op": "dense", "weights": rand(5, 3), "bias": rand(3)},
                  {"op": "activation", "activation": "relu"},
                  {"op": "batch_norm", **bn_params(3)}]
        fused, _ = fusion.fuse(layers)
        self.assertEqual([l["op"] for l in fused], ["dense", "batch_norm"])
        x = rand(2, 5)
        np.testing.assert_allclose(fusion.run(fused, x), fusion.run(layers, x), rtol=1e-6)

    def test_compose_disjoint_clamps(self):
        self.assertEqual(fusion._compose((-3.0, -2.0), "relu"), (0.0, 0.0))
        self.assertEqual(fusion._compose("relu", (-5.0, -1.0)), (-1.0, -1.0))
        self.assertEqual(fusion._compose("relu6", "relu"), (0.0, 6.0))


def _tf():
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except Exception:  # pragma: no cover - TF is optional
        return None


@unittest.skipUnless(fusion.available() and _tf() is not None, "needs TensorFlow")
class AgainstTensorFlowTest(unittest.TestCase):
    def test_keras_conv_bn_relu_model(self):
        tf = _tf()
        tf.keras.utils.set_random_seed(3)
        L = tf.keras.layers
        model = tf.keras.Sequential([
            tf.keras.Input((12, 12, 3)),
            L.Conv2D(8, 3, strides=2, padding="same", use_bias=False),
            L.BatchNormalization(), L.ReLU(max_value=6.0),
            L.DepthwiseConv2D(3, padding="same"), L.BatchNormalization(), L.ReLU(),
            L.Conv2D(6, 1), L.BatchNormalization(),
            L.MaxPooling2D(2), L.Flatten(), L.Dense(5, activation="relu"),
        ])
        # Give the batch norms non-trivial statistics.
        for l in model.layers:
            if isinstance(l, L.BatchNormalization):
                c = l.gamma.shape[0]
                l.set_weights([rand(c, lo=0.5, hi=1.5), rand(c, lo=-0.3, hi=0.3),
                               rand(c, lo=-0.3, hi=0.3), rand(c, lo=0.2, hi=1.2)])
        x = rand(3, 12, 12, 3)
        ref = model(x, training=False).numpy()
        layers = fusion.from_keras(model)
        fused, report = fusion.fuse(layers)
        self.assertEqual(len(fused), 6)
        np.testing.assert_allclose(fusion.run(layers, x), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(fusion.run(fused, x), ref, rtol=1e-4, atol=1e-4)

    def test_int8_add_matches_tflite(self):
        tf = _tf()
        rng = np.random.default_rng(5)
        for relu in (False, True):
            a_in, b_in = tf.keras.Input((4, 4, 8)), tf.keras.Input((4, 4, 8))
            y = tf.keras.layers.Add()([a_in, b_in])
            if relu:
                y = tf.keras.layers.ReLU()(y)
            m = tf.keras.Model([a_in, b_in], y)

            def rep():
                for _ in range(32):
                    yield [rng.uniform(-3, 4, (1, 4, 4, 8)).astype(np.float32),
                           rng.uniform(-9, 7, (1, 4, 4, 8)).astype(np.float32)]

            c = tf.lite.TFLiteConverter.from_keras_model(m)
            c.optimizations = [tf.lite.Optimize.DEFAULT]
            c.representative_dataset = rep
            c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            c.inference_input_type = tf.int8
            c.inference_output_type = tf.int8
            it = tf.lite.Interpreter(
                model_content=c.convert(),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
            it.allocate_tensors()
            ins = it.get_input_details()
            od = it.get_output_details()[0]
            xs = [rng.integers(-128, 128, (1, 4, 4, 8), dtype=np.int8) for _ in ins]
            for d, x in zip(ins, xs):
                it.set_tensor(d["index"], x)
            it.invoke()
            ref = it.get_tensor(od["index"])
            (sa, za), (sb, zb) = ins[0]["quantization"], ins[1]["quantization"]
            so, zo = od["quantization"]
            got = fusion.add_int8(xs[0], sa, za, xs[1], sb, zb, so, zo,
                                  "relu" if relu else None)
            np.testing.assert_array_equal(got, ref)


if __name__ == "__main__":
    unittest.main()
