"""PyTorch frontend: every supported operator, checked three ways.

1. The imported graph, run by the NumPy reference interpreter, against
   PyTorch's own output.
2. The generated C++, compiled and run, against the reference interpreter
   (float), and bit-exact against pTensor's bindings for int8.
3. Unsupported operators fail with a message that names them.

Skipped when torch is not installed.
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None

from ptensor.codegen import CodeGenerator, reference  # noqa: E402
from ptensor.codegen.frontends import load_model  # noqa: E402

import test_codegen  # noqa: E402  (compile helpers)


if torch is not None:
    class Fn(nn.Module):
        """Wrap a function of (module, x) as a module; extra submodules by keyword."""

        def __init__(self, fn, **mods):
            super().__init__()
            self.fn = fn
            for k, v in mods.items():
                setattr(self, k, v)

        def forward(self, x):
            return self.fn(self, x)

    def bn2d(c):
        m = nn.BatchNorm2d(c)
        with torch.no_grad():
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.uniform_(0.5, 1.5)
            m.bias.uniform_(-0.5, 0.5)
        return m

    def bn1d(c):
        m = nn.BatchNorm1d(c)
        with torch.no_grad():
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
        return m

    IMG = (1, 3, 9, 8)
    SEQ = (1, 5, 6)
    CH = (1, 4, 10)
    VEC = (1, 12)

    def cases():
        """(name, module, input shape)"""
        torch.manual_seed(0)
        C = []

        def add(name, mod, shape=IMG):
            C.append((name, mod, shape))

        # convolution
        add("conv_valid", nn.Conv2d(3, 4, 3))
        add("conv_pad1_same", nn.Conv2d(3, 4, 3, padding=1))
        add("conv_pad2_explicit", nn.Conv2d(3, 4, 3, padding=2))
        add("conv_s2_pad1", nn.Conv2d(3, 4, 3, stride=2, padding=1))
        add("conv_s2_pad0_k2", nn.Conv2d(3, 4, 2, stride=2))
        add("conv_rect", nn.Conv2d(3, 4, (3, 1), stride=(2, 1), padding=(1, 0)))
        add("conv_same_dil2", nn.Conv2d(3, 4, 3, padding="same", dilation=2))
        add("conv_dil2_pad1", nn.Conv2d(3, 4, 3, dilation=2, padding=1))
        add("conv_nobias", nn.Conv2d(3, 5, 1, bias=False))
        add("conv_groups", nn.Conv2d(4, 6, 3, padding=1, groups=2), (1, 4, 7, 7))
        add("conv_depthwise", nn.Conv2d(3, 3, 3, padding=1, groups=3))
        add("conv_depthwise_mult2_s2", nn.Conv2d(3, 6, 3, stride=2, padding=1, groups=3))
        add("conv1d", nn.Conv1d(4, 3, 3, padding=1), CH)
        add("conv1d_s2_valid", nn.Conv1d(4, 3, 3, stride=2), CH)
        add("conv1d_depthwise", nn.Conv1d(4, 4, 5, padding=2, groups=4), CH)
        # linear / matmul
        add("flatten_linear", Fn(lambda s, x: s.l(torch.flatten(F.relu(s.c(x)), 1)),
                                 c=nn.Conv2d(3, 4, 3), l=nn.Linear(4 * 7 * 6, 5)))
        add("view_linear", Fn(lambda s, x: s.l(x.view(x.size(0), -1)), l=nn.Linear(3 * 9 * 8, 4)))
        add("linear_seq", nn.Linear(6, 3), SEQ)
        add("linear_nobias", nn.Linear(12, 7, bias=False), VEC)
        add("matmul_act_act", Fn(lambda s, x: s.a(x) @ s.b(x).t(), a=nn.Linear(6, 4),
                                 b=nn.Linear(6, 4)), (5, 6))
        add("matmul_const_left", Fn(lambda s, x: s.w @ x, w=nn.Parameter(torch.randn(3, 5))),
            (5, 6))
        add("bmm", Fn(lambda s, x: torch.bmm(x, x.transpose(1, 2))), SEQ)
        add("attention", Fn(lambda s, x: torch.softmax(s.q(x) @ s.k(x).transpose(-2, -1) /
                                                       math.sqrt(4), dim=-1) @ s.v(x),
                            q=nn.Linear(6, 4), k=nn.Linear(6, 4), v=nn.Linear(6, 4)), SEQ)
        # pooling
        add("maxpool2", nn.MaxPool2d(2))
        add("maxpool3_s2_pad1", nn.MaxPool2d(3, 2, padding=1))
        add("maxpool3_s2_ceil", nn.MaxPool2d(3, 2, ceil_mode=True))
        add("avgpool2", nn.AvgPool2d(2))
        add("avgpool3_s2_pad1", nn.AvgPool2d(3, 2, padding=1))
        add("avgpool3_s1_pad1_exclude", nn.AvgPool2d(3, 1, padding=1, count_include_pad=False))
        add("adaptive_avg_1", nn.AdaptiveAvgPool2d(1))
        add("adaptive_avg_3x4", nn.AdaptiveAvgPool2d((3, 4)))
        add("adaptive_max_1", nn.AdaptiveMaxPool2d(1))
        add("maxpool1d", nn.MaxPool1d(2), CH)
        add("avgpool1d", nn.AvgPool1d(3, 2, padding=1), CH)
        add("adaptive_avg1d", nn.AdaptiveAvgPool1d(1), CH)
        add("mean_hw", Fn(lambda s, x: x.mean((2, 3))))
        add("mean_hw_keep", Fn(lambda s, x: x.mean((-1, -2), keepdim=True) * x))
        add("mean_c", Fn(lambda s, x: x.mean(1)))
        add("sum_w", Fn(lambda s, x: x.sum(3, keepdim=True)))
        # normalization
        add("batchnorm2d", bn2d(3))
        add("batchnorm1d_vec", Fn(lambda s, x: s.b(s.l(x)), l=nn.Linear(12, 5), b=bn1d(5)), VEC)
        add("batchnorm1d_seq", bn1d(4), CH)
        add("layernorm_last", nn.LayerNorm(6), SEQ)
        add("layernorm_chw", nn.LayerNorm([3, 9, 8]))
        add("groupnorm", nn.GroupNorm(2, 4), (1, 4, 5, 6))
        add("instancenorm", nn.InstanceNorm2d(3, affine=True))
        # activations
        for name, mod in [("relu", nn.ReLU()), ("relu6", nn.ReLU6()),
                          ("leaky_relu", nn.LeakyReLU(0.1)), ("elu", nn.ELU(0.7)),
                          ("celu", nn.CELU(0.5)), ("selu", nn.SELU()), ("gelu", nn.GELU()),
                          ("gelu_tanh", nn.GELU("tanh")), ("silu", nn.SiLU()), ("mish", nn.Mish()),
                          ("hardsigmoid", nn.Hardsigmoid()), ("hardswish", nn.Hardswish()),
                          ("hardtanh", nn.Hardtanh(-0.5, 2.0)), ("sigmoid", nn.Sigmoid()),
                          ("tanh", nn.Tanh()), ("softplus", nn.Softplus()),
                          ("softplus_beta2", nn.Softplus(beta=2.0)), ("softsign", nn.Softsign()),
                          ("logsigmoid", nn.LogSigmoid()), ("tanhshrink", nn.Tanhshrink()),
                          ("prelu_channel", nn.PReLU(3, 0.2)), ("prelu_one", nn.PReLU()),
                          ("softmax_c", nn.Softmax(dim=1)), ("softmax_w", nn.Softmax(dim=-1)),
                          ("log_softmax", nn.LogSoftmax(dim=1)), ("softmin", nn.Softmin(dim=1)),
                          ("clamp", Fn(lambda s, x: x.clamp(-0.3, 0.6))),
                          ("clamp_min", Fn(lambda s, x: torch.clamp(x, min=0.2))),
                          ("clamp_max", Fn(lambda s, x: torch.clamp(x, max=0.1)))]:
            add(name, mod)
        # elementwise math
        p = nn.Parameter(torch.randn(1, 3, 1, 1))
        add("add_sub_scalar", Fn(lambda s, x: (x + 1.5) - 0.25))
        add("rsub", Fn(lambda s, x: 2.0 - x))
        add("mul_div_scalar", Fn(lambda s, x: x * 3.0 / 2.0))
        add("mul_param", Fn(lambda s, x: x * s.p + s.p, p=p))
        add("add_alpha", Fn(lambda s, x: torch.add(x, F.relu(x), alpha=0.5)))
        add("sub_act", Fn(lambda s, x: torch.sub(x, torch.sigmoid(x), alpha=2)))
        add("div_act", Fn(lambda s, x: x / (x.abs() + 1.0)))
        add("reciprocal", Fn(lambda s, x: torch.reciprocal(x.abs() + 0.5)))
        add("max_min", Fn(lambda s, x: torch.maximum(x, x * 0.5) + torch.minimum(x, -x)))
        add("pow", Fn(lambda s, x: x ** 2 + x.abs() ** 0.5 + x ** 3))
        add("unary_math", Fn(lambda s, x: torch.sqrt(x.abs()) + torch.rsqrt(x.abs() + 1) +
                             torch.exp(x * 0.5) + torch.log(x.abs() + 1) + x.neg() +
                             torch.floor(x) + torch.ceil(x) + torch.round(x * 3) +
                             torch.sin(x) + torch.cos(x) + torch.square(x)))
        add("broadcast_vec", Fn(lambda s, x: x * s.v, v=nn.Parameter(torch.randn(8))))
        # shapes
        add("permute_flatten", Fn(lambda s, x: torch.flatten(x.permute(0, 2, 3, 1), 1)))
        add("transpose_hw", Fn(lambda s, x: x.transpose(2, 3).contiguous()))
        add("channel_shuffle", Fn(lambda s, x: x.view(1, 2, 3, 25).transpose(1, 2)
                                  .reshape(1, 6, 5, 5)), (1, 6, 5, 5))
        add("squeeze_unsqueeze", Fn(lambda s, x: s.l(F.adaptive_avg_pool2d(x, 1).squeeze(-1)
                                                     .squeeze(-1)).unsqueeze(1),
                                    l=nn.Linear(3, 4)))
        add("cat_channels", Fn(lambda s, x: torch.cat([s.a(x), s.b(x)], dim=1),
                               a=nn.Conv2d(3, 2, 1), b=nn.Conv2d(3, 3, 3, padding=1)))
        add("cat_width", Fn(lambda s, x: torch.cat([x, x * 2, x[:, :, :, :3]], dim=3)))
        add("stack", Fn(lambda s, x: torch.stack([x, x.sigmoid()], dim=1)), SEQ)
        add("slice_channels", Fn(lambda s, x: x[:, :2] * x[:, 1:]))
        add("slice_strided", Fn(lambda s, x: x[:, :, 1:8:3, ::2]))
        add("select", Fn(lambda s, x: x[:, 1] + x[:, :, 0].mean()))
        add("narrow_chunk_split", Fn(lambda s, x: x.narrow(3, 2, 4) + x.chunk(2, dim=3)[1] +
                                     x.split([1, 2], dim=1)[0][..., :4]))
        add("unbind", Fn(lambda s, x: sum(x.unbind(1))), SEQ)
        add("pad", Fn(lambda s, x: F.pad(x, (1, 2, 0, 1))))
        add("pad_value", Fn(lambda s, x: F.pad(x, (2, 0, 1, 1), value=-0.5)))
        add("pad1d", Fn(lambda s, x: F.pad(x, (3, 1))), CH)
        add("expand_const", Fn(lambda s, x: x + torch.ones(1, 3, 1, 1) * 0.5))
        # recurrent
        add("lstm", Fn(lambda s, x: s.r(x)[0], r=nn.LSTM(6, 5, batch_first=True)), SEQ)
        add("lstm_2layer_hn", Fn(lambda s, x: s.r(x)[1][0], r=nn.LSTM(6, 4, num_layers=2,
                                                                     batch_first=True)), SEQ)
        add("lstm_seq_first", Fn(lambda s, x: s.r(x)[0], r=nn.LSTM(6, 3)), (5, 1, 6))
        add("gru", Fn(lambda s, x: s.r(x)[0], r=nn.GRU(6, 5, batch_first=True)), SEQ)
        add("gru_hn_nobias", Fn(lambda s, x: s.r(x)[1], r=nn.GRU(6, 4, bias=False,
                                                                batch_first=True)), SEQ)
        # attention, batching, tiling, reductions, upsampling
        add("bmm_batched", Fn(lambda s, x: torch.bmm(x, x.transpose(1, 2)) * 0.1), (3, 4, 5))
        add("matmul_broadcast_const", Fn(lambda s, x: x @ s.w, w=nn.Parameter(torch.randn(5, 2))),
            (3, 4, 5))
        add("matmul_vec", Fn(lambda s, x: x @ s.w, w=nn.Parameter(torch.randn(6))), SEQ)
        add("sdpa_heads_causal", Fn(lambda s, x: F.scaled_dot_product_attention(
            *[t.view(1, 5, 2, 3).transpose(1, 2) for t in s.qkv(x).chunk(3, dim=-1)],
            is_causal=True).transpose(1, 2).reshape(1, 5, 6), qkv=nn.Linear(6, 18)), SEQ)
        add("multihead_attention", Fn(lambda s, x: s.m(x, x, x, need_weights=False)[0],
                                      m=nn.MultiheadAttention(6, 2, batch_first=True)), SEQ)
        add("transformer_encoder_layer", nn.TransformerEncoderLayer(6, 2, 12, dropout=0.0,
                                                                    batch_first=True), SEQ)
        add("transformer_gelu_norm_first", nn.TransformerEncoderLayer(
            6, 3, 8, dropout=0.0, activation="gelu", norm_first=True, batch_first=True), SEQ)
        add("repeat", Fn(lambda s, x: x.repeat(1, 2, 1, 3)), (1, 2, 3, 2))
        add("expand", Fn(lambda s, x: x.mean(1, keepdim=True).expand(1, 3, 9, 8) - x))
        add("amax_amin", Fn(lambda s, x: x.amax((2, 3)) - torch.amin(x, dim=(2, 3))))
        add("max_dim", Fn(lambda s, x: x.max(dim=3)[0].max(dim=2, keepdim=True)[0]))
        add("var_std", Fn(lambda s, x: x.var(dim=1) + x.std(dim=(2, 3), keepdim=True).mean(1)
                          + torch.var(x, dim=1, unbiased=False)))
        add("upsample2x", nn.Upsample(scale_factor=2))
        add("upsample_3x2", nn.Upsample(size=(27, 16), mode="nearest"))
        add("upsample1d", nn.Upsample(scale_factor=3), CH)
        add("unflatten", Fn(lambda s, x: torch.flatten(x, 1).unflatten(1, (3, 72)).sum(1)))
        # whole models
        add("resnet_block", Fn(lambda s, x: F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sc(x)),
                               c1=nn.Conv2d(3, 8, 3, padding=1, bias=False), b1=bn2d(8),
                               c2=nn.Conv2d(8, 8, 3, padding=1, bias=False), b2=bn2d(8),
                               sc=nn.Conv2d(3, 8, 1)))

        def mbv3(s, x):
            y = F.hardswish(s.b1(s.pw(x)))
            y = F.hardswish(s.b2(s.dw(y)))
            se = F.hardsigmoid(s.f2(F.relu(s.f1(F.adaptive_avg_pool2d(y, 1).flatten(1)))))
            y = y * se.view(1, -1, 1, 1)
            return s.head(torch.flatten(F.adaptive_avg_pool2d(s.proj(y), 1), 1))
        add("mobilenetv3_block", Fn(mbv3, pw=nn.Conv2d(3, 16, 1, bias=False), b1=bn2d(16),
                                    dw=nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16,
                                                 bias=False), b2=bn2d(16), f1=nn.Linear(16, 4),
                                    f2=nn.Linear(4, 16), proj=nn.Conv2d(16, 8, 1),
                                    head=nn.Linear(8, 10)))
        add("mlp", nn.Sequential(nn.Flatten(), nn.Linear(3 * 9 * 8, 32), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(32, 10), nn.LogSoftmax(dim=1)))
        add("kws_conv1d_gru", Fn(lambda s, x: s.fc(s.rnn(F.relu(s.c(x)).transpose(1, 2))[0][:, -1]),
                                 c=nn.Conv1d(4, 8, 3, padding=1), rnn=nn.GRU(8, 6, batch_first=True),
                                 fc=nn.Linear(6, 3)), CH)
        return C

    def to_ir(x):
        return x.transpose(0, 2, 3, 1) if x.ndim == 4 else x

    def from_ir(y, shape):
        if len(shape) == 4:
            return y.reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
        return y.reshape(shape)


@unittest.skipIf(torch is None, "needs torch")
class PyTorchImportTest(unittest.TestCase):
    """Imported graph (NumPy reference) == PyTorch, for every case."""

    def test_every_case_matches_pytorch(self):
        rng = np.random.default_rng(3)
        for name, mod, shape in cases():
            with self.subTest(name):
                mod.eval()
                x = rng.normal(size=shape).astype(np.float32)
                g = load_model(mod, example_input=torch.from_numpy(x))
                with torch.no_grad():
                    want = mod(torch.from_numpy(x)).numpy()
                got = reference.run(g, to_ir(x))[0]
                got = from_ir(got, want.shape)
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-5, err_msg=name)

    def test_flatten_into_linear_permutes_rows_not_data(self):
        mod = Fn(lambda s, x: s.l(torch.flatten(s.c(x), 1)), c=nn.Conv2d(3, 4, 3),
                 l=nn.Linear(4 * 7 * 6, 5)).eval()
        g = load_model(mod, example_input=torch.zeros(IMG))
        self.assertNotIn("transpose", g.ops_used())

    def test_nchw_io(self):
        mod = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()).eval()
        x = torch.randn(IMG)
        g = load_model(mod, example_input=x, channels_last_io=False)
        self.assertEqual(g.tensors[g.inputs[0]].shape, IMG)
        got = reference.run(g, x.numpy())[0]
        with torch.no_grad():
            np.testing.assert_allclose(got, mod(x).numpy(), atol=2e-5)

    def test_exported_program_and_file(self):
        mod = nn.Sequential(nn.Linear(12, 4), nn.Tanh()).eval()
        x = torch.randn(VEC)
        ep = torch.export.export(mod, (x,))
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "tiny.pt2")
            torch.export.save(ep, path)
            for src in (ep, path):
                g = load_model(src)
                with torch.no_grad():
                    np.testing.assert_allclose(reference.run(g, x.numpy())[0], mod(x).numpy(),
                                               atol=2e-6)
        finally:
            shutil.rmtree(d)

    def test_unsupported_operators_name_themselves(self):
        for mod, what in [(nn.ConvTranspose2d(3, 2, 2), "conv_transpose2d"),
                          (nn.Upsample(scale_factor=2, mode="bilinear"), "upsample_bilinear2d"),
                          (Fn(lambda s, x: F.pad(x, (1, 1, 1, 1), mode="reflect")), "reflect"),
                          (nn.LSTM(9, 3, bidirectional=True), "bidirectional")]:
            with self.subTest(what):
                shape = (1, 3, 9) if isinstance(mod, nn.LSTM) else IMG
                with self.assertRaises(NotImplementedError) as cm:
                    load_model(mod.eval(), example_input=torch.randn(shape))
                self.assertIn(what.split()[0], str(cm.exception).lower())

    def test_module_needs_example_input(self):
        with self.assertRaises(ValueError):
            load_model(nn.ReLU())


@unittest.skipIf(torch is None, "needs torch")
@unittest.skipUnless(test_codegen._build_dir() and (shutil.which("g++") or shutil.which("c++")),
                     "needs a C++ compiler and a CMake build of pTensor")
class PyTorchCompileTest(unittest.TestCase):
    """Generated C++ == reference interpreter, for every case."""
    @classmethod
    def setUpClass(cls):
        cls.build = test_codegen._build_dir()
        cls.cache = test_codegen._cache(cls.build)
        cls.tmp = tempfile.mkdtemp(prefix="ptensor_pytorch_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    has = test_codegen.CompileTest.has
    run_model = test_codegen.CompileTest.run_model

    def check_case(self, name, mod, shape, **opts):
        rng = np.random.default_rng(4)
        xs = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
        gen = CodeGenerator(mod.eval(), name="pt_" + name, example_input=torch.from_numpy(xs[0]),
                            **opts)
        y = self.run_model(gen, [to_ir(x) for x in xs])
        exp = np.stack([gen.reference(to_ir(x))[0].reshape(-1) for x in xs])
        np.testing.assert_allclose(y, exp, rtol=2e-4, atol=5e-5, err_msg=name)
        with torch.no_grad():
            want = np.stack([mod(torch.from_numpy(x)).numpy() for x in xs])
        got = np.stack([from_ir(yi, want.shape[1:]) for yi in y])
        np.testing.assert_allclose(got, want, rtol=5e-4, atol=1e-4, err_msg=name)

    def test_every_case_compiles_and_matches(self):
        if not self.has("PTENSOR_WITH_FUSION"):
            self.skipTest("built without fusion")
        for name, mod, shape in cases():
            with self.subTest(name):
                self.check_case(name, mod, shape)

    def test_unfused_dynamic(self):
        for name, mod, shape in cases():
            if name in ("resnet_block", "mobilenetv3_block", "kws_conv1d_gru", "attention",
                        "conv_same_dil2", "conv_dil2_pad1"):
                with self.subTest(name):
                    if name.startswith("conv_") and "dil" in name:
                        continue  # the reference conv has no dilation; needs fuse=True
                    self.check_case(name + "_unfused", mod, shape, fuse=False, memory="dynamic")

    def test_int8_cnn_matches_bindings(self):
        if not self.has("PTENSOR_WITH_QUANTIZED_OPS"):
            self.skipTest("built without the int8 operators")
        torch.manual_seed(1)
        mod = Fn(lambda s, x: torch.softmax(s.l(torch.flatten(F.max_pool2d(
            F.relu(s.b(s.c(x))), 3, 2, padding=1), 1)), dim=1),
            c=nn.Conv2d(3, 6, 3, padding=1), b=bn2d(6), l=nn.Linear(6 * 5 * 4, 5)).eval()
        rng = np.random.default_rng(6)
        cal = [rng.uniform(0, 1, IMG).astype(np.float32) for _ in range(16)]
        gen = CodeGenerator(mod, name="pt_int8", example_input=torch.from_numpy(cal[0]),
                            activations="int8", calibration_data=[to_ir(c) for c in cal],
                            input_scale=1 / 255.0, input_zero_point=-128)
        self.assertIn("pad", gen.graph.ops_used())
        xs = [np.clip(np.rint(to_ir(c) * 255) - 128, -128, 127).astype(np.int8) for c in cal[:5]]
        y = self.run_model(gen, xs)
        exp = np.stack([reference.run_ptensor_int8(gen.graph, xi)[0].reshape(-1) for xi in xs])
        np.testing.assert_array_equal(y, exp)
        with torch.no_grad():
            want = np.stack([mod(torch.from_numpy(c)).numpy().reshape(-1) for c in cal[:5]])
        q = gen.graph.tensors[gen.graph.outputs[0]].quant
        got = (y.astype(np.float32) - q.zero_point[0]) * q.scale[0]
        self.assertLess(np.abs(got - want).max(), 0.05)


if __name__ == "__main__":
    unittest.main()
