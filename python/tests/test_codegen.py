"""ptensor.codegen: graph passes, memory planning, the C emitter and (when a
compiler and the static libraries are available) generated code compiled and
run against the NumPy reference / pTensor's own kernels.

Needs only NumPy. The TFLite frontend test runs when the 'tflite' package is
installed; examples/codegen/verify_models.py covers Keras / TF end to end.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from ptensor import codegen
from ptensor.codegen import CodeGenerator, ir, passes, reference

RNG = np.random.default_rng(5)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def rand(*shape, scale=0.5):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def small_cnn(bn=True, residual=True):
    """conv(relu) -> [bn] -> maxpool -> depthwise -> [add residual, relu] ->
    conv -> flatten -> dense(relu) -> dense -> softmax, on 8x8x2 input."""
    g = ir.Graph("small")
    x = g.add_tensor("x", (1, 8, 8, 2))
    g.inputs.append(x.name)
    c1 = g.add_tensor("c1", (1, 8, 8, 4))
    g.add_node("conv2d", [x, g.add_const("w1", rand(4, 3, 3, 2)), g.add_const("b1", rand(4))], [c1],
               {"stride": (1, 1), "padding": "SAME", "dilation": (1, 1), "activation": "relu"}, "conv1")
    cur = c1
    if bn:
        b = g.add_tensor("bn", (1, 8, 8, 4))
        g.add_node("batch_norm", [cur, g.add_const("m", rand(4)), g.add_const("v", np.abs(rand(4)) + 0.5),
                                  g.add_const("ga", rand(4) + 1), g.add_const("be", rand(4))],
                   [b], {"epsilon": 1e-3}, "bn")
        cur = b
    p = g.add_tensor("p", (1, 4, 4, 4))
    g.add_node("max_pool2d", [cur], [p], {"kernel": (2, 2), "stride": (2, 2), "padding": "VALID"}, "pool")
    d = g.add_tensor("d", (1, 4, 4, 4))
    g.add_node("depthwise_conv2d", [p, g.add_const("dw", rand(1, 3, 3, 4)), g.add_const("db", rand(4))],
               [d], {"stride": (1, 1), "padding": "SAME", "dilation": (1, 1), "depth_multiplier": 1,
                     "activation": None}, "dw")
    cur = d
    if residual:
        s = g.add_tensor("s", (1, 4, 4, 4))
        g.add_node("add", [p, d], [s], {"activation": None}, "add")
        r = g.add_tensor("r", (1, 4, 4, 4))
        g.add_node("activation", [s], [r], {"kind": "relu6"}, "relu6")
        cur = r
    c2 = g.add_tensor("c2", (1, 4, 4, 6))
    g.add_node("conv2d", [cur, g.add_const("w2", rand(6, 1, 1, 4)), g.add_const("b2", rand(6))], [c2],
               {"stride": (1, 1), "padding": "VALID", "dilation": (1, 1), "activation": "relu"}, "conv2")
    f = g.add_tensor("f", (1, 96))
    g.add_node("reshape", [c2], [f], name="flatten")
    h = g.add_tensor("h", (1, 12))
    g.add_node("dense", [f, g.add_const("w3", rand(96, 12, scale=0.2)), g.add_const("b3", rand(12))], [h],
               {"activation": "relu"}, "fc1")
    o = g.add_tensor("o", (1, 5))
    g.add_node("dense", [h, g.add_const("w4", rand(12, 5)), g.add_const("b4", rand(5))], [o],
               {"activation": None}, "fc2")
    y = g.add_tensor("y", (1, 5))
    g.add_node("softmax", [o], [y], {"beta": 1.0}, "softmax")
    g.outputs.append(y.name)
    g.source = "test"
    return g.validate()


def inputs(n=6):
    return [RNG.uniform(0, 1, (1, 8, 8, 2)).astype(np.float32) for _ in range(n)]


class PassTest(unittest.TestCase):
    def assert_same(self, g1, g2, tol=1e-5):
        for x in inputs():
            a, b = reference.run(g1, x)[0], reference.run(g2, x)[0]
            np.testing.assert_allclose(a, b, atol=tol)

    def test_fuse_folds_bn_and_activations(self):
        g = small_cnn()
        f = passes.fuse(g)
        ops = [n.op for n in f.nodes]
        self.assertNotIn("activation", ops)  # relu6 after add was absorbed
        self.assertIn("batch_norm", ops)     # bn after a relu cannot fold
        self.assert_same(g, f)
        g2 = small_cnn()
        g2.nodes[0].attrs["activation"] = None  # conv -> bn now folds
        f2 = passes.fuse(g2)
        self.assertNotIn("batch_norm", [n.op for n in f2.nodes])
        self.assert_same(g2, f2)

    def test_unfuse_splits_activations(self):
        g = small_cnn()
        u = passes.unfuse(g)
        self.assertTrue(all(n.attrs.get("activation") is None for n in u.nodes))
        self.assert_same(g, u)

    def test_prune_removes_dead_channels_exactly(self):
        g = small_cnn(bn=False, residual=False)
        # conv2 feeds the flatten + dense: kill 2 of its 6 filters
        w2, b2 = g.tensors["w2"], g.tensors["b2"]
        w2.data[[1, 4]] = 0
        b2.data[[1, 4]] = -1.0  # relu(-1) = 0: the channels are dead
        # fc1 units feeding fc2: kill 3 of 12
        w3, b3 = g.tensors["w3"], g.tensors["b3"]
        w3.data[:, [0, 5, 7]] = 0
        b3.data[[0, 5, 7]] = -1.0
        p = passes.prune(g, {"conv2": 2 / 6.0, "fc1": 3 / 12.0})
        self.assertEqual(p.tensors[p.outputs[0]].shape, (1, 5))
        shapes = {n.name: p.tensors[n.outputs[0]].shape for n in p.nodes}
        self.assertEqual(shapes["conv2"], (1, 4, 4, 4))
        self.assertEqual(shapes["flatten"], (1, 64))
        self.assertEqual(shapes["fc1"], (1, 9))
        self.assert_same(g, p)

    def test_prune_skips_residual_and_outputs(self):
        g = small_cnn()
        p = passes.prune(g, 0.5)
        # conv1 feeds bn -> pool -> add (residual): not prunable; fc2 is the output
        self.assertEqual(p.tensors[p.nodes[0].outputs[0]].shape[-1], 4)
        self.assertEqual(p.tensors[p.outputs[0]].shape, (1, 5))

    def test_encode_weights_decodes_like_device(self):
        g = passes.fuse(small_cnn())
        e = passes.encode_weights(g, "uint4", min_size=16)
        enc = [t for t in e.tensors.values() if t.encoding is not None]
        self.assertTrue(enc)
        self.assertTrue(all(t.encoding.describe() == "uint4" for t in enc))
        x = inputs(1)[0]
        a, b = reference.run(g, x)[0], reference.run(e, x)[0]
        self.assertLess(np.abs(a - b).max(), 0.2)
        self.assertFalse(np.array_equal(a, b))

    def test_quantize_int8(self):
        g = passes.fuse(small_cnn(bn=False))
        q = passes.quantize_int8(g, inputs(20), input_scale=1 / 255.0, input_zero_point=-128)
        self.assertTrue(q.is_quantized())
        for n in q.nodes:
            if n.op in ir.LINEAR_OPS:
                self.assertEqual(q.tensors[n.inputs[1]].dtype, "int8")
                self.assertEqual(q.tensors[n.inputs[2]].dtype, "int32")
        self.assertEqual(q.tensors[q.outputs[0]].quant.zero_point[0], -128)
        with self.assertRaises(NotImplementedError):
            passes.quantize_int8(small_cnn(), inputs(2))  # unfolded batch norm

    def test_plan_memory_never_overlaps_live_tensors(self):
        for g in (small_cnn(), passes.fuse(small_cnn()), passes.unfuse(small_cnn())):
            plan = passes.plan_memory(g, alignment=4)
            acts = g.activations()
            n = len(g.nodes)
            life = {}
            for t in acts:
                first = -1 if t in g.inputs else min(i for i, nd in enumerate(g.nodes) if t in nd.outputs)
                last = n if t in g.outputs else max([i for i, nd in enumerate(g.nodes) if t in nd.inputs] + [first])
                life[t] = (first, last)

            def root(t):
                while t in plan.shared:
                    t = plan.shared[t]
                return t
            for i, a in enumerate(acts):
                for b in acts[i + 1:]:
                    if root(a) == root(b):
                        self.assertEqual(plan.offsets[a], plan.offsets[b])
                        continue
                    (fa, la), (fb, lb) = life[a], life[b]
                    if la < fb or lb < fa:
                        continue
                    sa, sb = g.tensors[a].nbytes, g.tensors[b].nbytes
                    oa, ob = plan.offsets[a], plan.offsets[b]
                    self.assertTrue(oa + sa <= ob or ob + sb <= oa, "%s and %s overlap" % (a, b))
                    self.assertLessEqual(max(oa + sa, ob + sb), plan.arena_bytes)
            self.assertLessEqual(plan.arena_bytes, plan.naive_bytes)
            self.assertIn("f", plan.shared)  # the flatten is a view


class EmitterTest(unittest.TestCase):
    def test_features_and_includes(self):
        F, H, P = "PTENSOR_WITH_FUSION", "PTENSOR_WITH_HOUGH", "PTENSOR_WITH_POLY_QUANT"
        cases = [  # options, features needed, features not needed, text in the .cpp
            (dict(), {F}, {H, P}, "FusedConv2dOperator"),
            (dict(fuse=False), set(), {F, H, P}, "ReferenceOperators::Conv2dOperator"),
            # Hough falls back to uniform codes on tensors this small, so the
            # model must not ask for the Hough feature.
            (dict(weights="hough4", encode_min_size=16), {F}, {H, P}, "PackedRomTensor"),
            (dict(weights="cubic4", encode_min_size=16, fuse=False), {P}, {F, H}, "PolyRomTensor"),
            (dict(weights="uint5", encode_min_size=16, weights_in_ram=True), {F}, {H, P}, "_ram[i]"),
        ]
        for opts, need, not_need, marker in cases:
            gen = CodeGenerator(small_cnn(), name="t", fast_kernels=False, **opts)
            files = gen.emit()
            self.assertEqual(sorted(files), ["t.cmake", "t.cpp", "t.h"])
            src = files["t.cpp"]
            self.assertIn(marker, src, opts)
            self.assertTrue(need <= set(gen.features), (opts, gen.features))
            self.assertFalse(not_need & set(gen.features), (opts, gen.features))
            self.assertNotIn("PTENSOR_WITH_QUANTIZED_OPS", gen.features)
            for f in gen.features:
                self.assertIn('set(%s ON CACHE BOOL "needed by t" FORCE)' % f, files["t.cmake"])
                self.assertIn("#if !%s" % f, src)
            self.assertNotIn("pTensor.h", src)  # only the headers it needs

    def test_int8_uses_tflm_ops(self):
        g = passes.fuse(small_cnn(bn=False))
        gen = CodeGenerator(g, name="q", activations="int8", calibration_data=inputs(10))
        src = gen.emit()["q.cpp"]
        self.assertIn("TflmSymQuantOps::Conv2dOperator<int8_t>", src)
        self.assertIn("TflmSymQuantOps::QuantizedAddOperator<int8_t>", src)
        self.assertIn("PTENSOR_WITH_QUANTIZED_OPS", gen.features)
        hdr = gen.files["q.h"]
        self.assertRegex(hdr, r"#define Q_IN0_SCALE")
        self.assertIn("int q_invoke(const int8_t* in0, int8_t* out0);", hdr)

    def test_unsupported_reports_clearly(self):
        g = small_cnn()
        with self.assertRaises(ValueError):
            CodeGenerator(g, activations="int8")  # no calibration data
        with self.assertRaises(ValueError):
            codegen.load_model(42)
        with self.assertRaises(codegen.FrontendUnavailable):
            codegen.load_model("x", framework="pytorch")

    def test_custom_frontend(self):
        @codegen.register_frontend
        class Mine(codegen.Frontend):
            name = "unit-test"

            def accepts(self, source):
                return source == "mine"

            def load(self, source, **options):
                return small_cnn()
        self.assertIn("unit-test", codegen.available_frontends())
        self.assertEqual(codegen.load_model("mine", framework="unit-test").name, "small")


# ---------------------------------------------------------------------------
# Compile and run generated code (needs g++ and the static libraries)
# ---------------------------------------------------------------------------

def _build_dir():
    lib = os.environ.get("PTENSOR_LIBRARY")
    cands = [os.path.dirname(os.path.dirname(os.path.dirname(lib)))] if lib else []
    cands += [os.path.join(REPO, d) for d in sorted(os.listdir(REPO)) if d.startswith("build")]
    for b in cands:
        if os.path.exists(os.path.join(b, "src", "pTensor", "libptensor.a")):
            return b
    return None


def _cache(build):
    try:
        text = open(os.path.join(build, "CMakeCache.txt")).read()
    except OSError:
        return {}
    return dict(re.findall(r"^([A-Za-z_0-9]+):[A-Z]+=(.*)$", text, re.M))


HARNESS = r"""
#include <cstdio>
#include <vector>
#include "%(name)s.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); FILE* g = fopen(argv[2], "wb");
  std::vector<%(in_t)s> x(%(N)s_IN0_SIZE); std::vector<%(out_t)s> y(%(N)s_OUT0_SIZE);
  while (fread(x.data(), sizeof(x[0]), x.size(), f) == x.size()) {
    if (%(name)s_invoke(x.data(), y.data())) return 1;
    fwrite(y.data(), sizeof(y[0]), y.size(), g);
  }
  return 0;
}
"""


@unittest.skipUnless(_build_dir() and (shutil.which("g++") or shutil.which("c++")),
                     "needs a C++ compiler and a CMake build of pTensor")
class CompileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _build_dir()
        cls.cache = _cache(cls.build)
        cls.tmp = tempfile.mkdtemp(prefix="ptensor_codegen_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def has(self, feature):
        return self.cache.get(feature, "ON").upper() in ("ON", "1", "TRUE")

    def run_model(self, gen, x):
        d = os.path.join(self.tmp, gen.name)
        gen.generate(d)
        g = gen.graph
        ct = {"float32": "float", "int8": "int8_t"}
        ti, to = g.tensors[g.inputs[0]], g.tensors[g.outputs[0]]
        with open(os.path.join(d, "main.cpp"), "w") as f:
            f.write(HARNESS % {"name": gen.name, "N": gen.name.upper(), "in_t": ct[ti.dtype],
                               "out_t": ct[to.dtype]})
        cxx = self.cache.get("CMAKE_CXX_COMPILER") or shutil.which("g++") or "c++"
        flags = ["-std=c++11", "-O1", "-fno-rtti"]
        if self.cache.get("PTENSOR_SANITIZE", "OFF").upper() == "ON":
            flags += ["-fsanitize=address,undefined"]
        inc = []
        for sub in ("src", "src/pTensor", "src/pTensor/core", "src/pTensor/allocators",
                    "src/pTensor/tensors", "src/pTensor/ops", "src/pTensor/ops/reference",
                    "src/pTensor/ops/tflm", "src/pTensor/errorHandlers"):
            inc += ["-I", os.path.join(REPO, sub)]
        inc += ["-I", os.path.join(self.build, "src", "pTensor", "core", "generated")]
        exe = os.path.join(d, "run")
        cmd = [cxx] + flags + inc + [os.path.join(d, gen.name + ".cpp"), os.path.join(d, "main.cpp"),
                                     os.path.join(self.build, "src", "pTensor", "libptensor.a"),
                                     os.path.join(self.build, "src", "pTensor", "core", "libptensor_core.a"),
                                     "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        np.concatenate([np.asarray(xi).reshape(-1) for xi in x]).astype(
            {"float32": np.float32, "int8": np.int8}[ti.dtype]).tofile(os.path.join(d, "in.bin"))
        r = subprocess.run([exe, os.path.join(d, "in.bin"), os.path.join(d, "out.bin")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return np.fromfile(os.path.join(d, "out.bin"),
                           {"float32": np.float32, "int8": np.int8}[to.dtype]).reshape(len(x), -1)

    count = 0

    def check_float(self, **opts):
        CompileTest.count += 1
        gen = CodeGenerator(small_cnn(), name="m%d" % CompileTest.count, **opts)
        xs = inputs(5)
        y = self.run_model(gen, xs)
        exp = np.stack([gen.reference(xi)[0].reshape(-1) for xi in xs])
        np.testing.assert_allclose(y, exp, atol=2e-5)

    def test_float_fused_planned(self):
        if not self.has("PTENSOR_WITH_FUSION"):
            self.skipTest("built without fusion")
        self.check_float()

    def test_float_unfused_dynamic(self):
        self.check_float(fuse=False, memory="dynamic")

    def test_packed_weights_flash_and_ram(self):
        self.check_float(fuse=False, weights="uint4", encode_min_size=16)
        self.check_float(fuse=False, weights="int8", encode_min_size=16, weights_in_ram=True)

    def test_hough_weights(self):
        if not self.has("PTENSOR_WITH_HOUGH"):
            self.skipTest("built without Hough")
        g = ir.Graph("mlp")
        x = g.add_tensor("x", (1, 8, 8, 1))
        g.inputs.append(x.name)
        f = g.add_tensor("f", (1, 64))
        g.add_node("reshape", [x], [f])
        h = g.add_tensor("h", (1, 64))
        g.add_node("dense", [f, g.add_const("w1", rand(64, 64, scale=0.2)), g.add_const("b1", rand(64))],
                   [h], {"activation": "relu"}, "fc1")
        y = g.add_tensor("y", (1, 5))
        g.add_node("dense", [h, g.add_const("w2", rand(64, 5)), g.add_const("b2", rand(5))], [y],
                   {"activation": None}, "fc2")
        g.outputs.append(y.name)
        for in_ram in (False, True):
            gen = CodeGenerator(g, name="mlp%d" % in_ram, weights="hough4", weights_in_ram=in_ram,
                                fuse=False)
            self.assertIn("PTENSOR_WITH_HOUGH", gen.emit() and gen.features)
            xs = [RNG.uniform(0, 1, (1, 8, 8, 1)).astype(np.float32) for _ in range(4)]
            y_c = self.run_model(gen, xs)
            exp = np.stack([gen.reference(xi)[0].reshape(-1) for xi in xs])
            np.testing.assert_allclose(y_c, exp, atol=2e-5)

    def test_poly_weights(self):
        if self.has("PTENSOR_WITH_POLY_QUANT"):
            self.check_float(fuse=False, weights="cubic4", encode_min_size=16)

    def test_int8_matches_bindings(self):
        if not (self.has("PTENSOR_WITH_QUANTIZED_OPS") and self.has("PTENSOR_WITH_FUSION")):
            self.skipTest("built without the int8 operators")
        g = small_cnn(bn=False)
        g.nodes = [n for n in g.nodes if n.op != "depthwise_conv2d"]  # no binding for int8 depthwise
        add = [n for n in g.nodes if n.op == "add"][0]
        add.inputs = ["p", "p"]
        g.remove_unused()
        g.validate()
        gen = CodeGenerator(g, name="m_int8", activations="int8", calibration_data=inputs(20),
                            input_scale=1 / 255.0, input_zero_point=-128)
        q = gen.graph
        xs = [np.clip(np.rint(xi * 255) - 128, -128, 127).astype(np.int8) for xi in inputs(5)]
        y = self.run_model(gen, xs)
        exp = np.stack([reference.run_ptensor_int8(q, xi)[0].reshape(-1) for xi in xs])
        np.testing.assert_array_equal(y, exp)

    def test_int8_avgpool_sigmoid_tanh_match_bindings(self):
        if not self.has("PTENSOR_WITH_QUANTIZED_OPS"):
            self.skipTest("built without the int8 operators")
        # conv -> avg pool (SAME, ragged border) -> tanh -> conv -> sigmoid ->
        # avg pool (VALID) -> flatten -> dense
        g = ir.Graph("act")
        x = g.add_tensor("x", (1, 7, 7, 2))
        g.inputs.append(x.name)
        c1 = g.add_tensor("c1", (1, 7, 7, 4))
        g.add_node("conv2d", [x, g.add_const("w1", rand(4, 3, 3, 2)), g.add_const("b1", rand(4))], [c1],
                   {"stride": (1, 1), "padding": "SAME", "dilation": (1, 1), "activation": None}, "conv1")
        p1 = g.add_tensor("p1", (1, 4, 4, 4))
        g.add_node("avg_pool2d", [c1], [p1], {"kernel": (3, 3), "stride": (2, 2), "padding": "SAME"}, "pool1")
        t = g.add_tensor("t", (1, 4, 4, 4))
        g.add_node("activation", [p1], [t], {"kind": "tanh"}, "tanh")
        c2 = g.add_tensor("c2", (1, 4, 4, 3))
        g.add_node("conv2d", [t, g.add_const("w2", rand(3, 1, 1, 4)), g.add_const("b2", rand(3))], [c2],
                   {"stride": (1, 1), "padding": "VALID", "dilation": (1, 1), "activation": None}, "conv2")
        s = g.add_tensor("s", (1, 4, 4, 3))
        g.add_node("activation", [c2], [s], {"kind": "sigmoid"}, "sigmoid")
        p2 = g.add_tensor("p2", (1, 2, 2, 3))
        g.add_node("avg_pool2d", [s], [p2], {"kernel": (2, 2), "stride": (2, 2), "padding": "VALID"}, "pool2")
        f = g.add_tensor("f", (1, 12))
        g.add_node("reshape", [p2], [f], name="flatten")
        y = g.add_tensor("y", (1, 3))
        g.add_node("dense", [f, g.add_const("w3", rand(12, 3)), g.add_const("b3", rand(3))], [y],
                   {"activation": None}, "fc")
        g.outputs.append(y.name)
        g.validate()
        cal = [RNG.uniform(0, 1, (1, 7, 7, 2)).astype(np.float32) for _ in range(20)]
        gen = CodeGenerator(g, name="m_act8", activations="int8", calibration_data=cal,
                            input_scale=1 / 255.0, input_zero_point=-128)
        src = gen.emit()["m_act8.cpp"]
        self.assertIn("AvgPoolOperator<int8_t>", src)
        self.assertIn("LogisticOperator<int8_t>", src)
        self.assertIn("TanhOperator<int8_t>", src)
        q = gen.graph
        # TFLite's fixed output quantization
        self.assertEqual(q.tensors["s"].quant.zero_point[0], -128)
        self.assertAlmostEqual(float(q.tensors["s"].quant.scale[0]), 1 / 256.0)
        self.assertEqual(q.tensors["t"].quant.zero_point[0], 0)
        self.assertAlmostEqual(float(q.tensors["t"].quant.scale[0]), 1 / 128.0)
        self.assertEqual(q.tensors["p1"].quant.scale[0], q.tensors["c1"].quant.scale[0])
        xs = [np.clip(np.rint(xi * 255) - 128, -128, 127).astype(np.int8) for xi in cal[:6]]
        y_c = self.run_model(gen, xs)
        exp = np.stack([reference.run_ptensor_int8(q, xi)[0].reshape(-1) for xi in xs])
        np.testing.assert_array_equal(y_c, exp)
        # and close to the float model
        yf = np.stack([reference.run(g, xi)[0].reshape(-1) for xi in cal[:6]])
        yq = (exp.astype(np.float32) - q.tensors["y"].quant.zero_point[0]) * q.tensors["y"].quant.scale[0]
        self.assertLess(np.abs(yq - yf).max(), 0.1)


class TFLiteFrontendTest(unittest.TestCase):
    def test_mnist_int8(self):
        try:
            import tflite  # noqa: F401
        except ImportError:
            self.skipTest("needs the tflite package")
        g = codegen.load_model(os.path.join(REPO, "examples", "mnist", "mnist_int8.tflite"))
        self.assertTrue(g.is_quantized())
        self.assertEqual([n.op for n in g.nodes],
                         ["conv2d", "max_pool2d", "conv2d", "max_pool2d", "reshape", "dense",
                          "dense", "softmax"])
        self.assertEqual(g.tensors[g.nodes[5].inputs[1]].shape, (784, 32))  # [in, out]
        plan = passes.plan_memory(g)
        self.assertEqual(plan.arena_bytes, 7840)  # same as examples/memory_plan


if __name__ == "__main__":
    unittest.main()
