"""ONNX (placeholder).

Registered so ``.onnx`` files are recognised and fail with a clear message.
Until this importer is written, convert with ``onnx2tf -i model.onnx`` (which
writes TFLite) and pass the .tflite file.

Writing it: ``onnx.load`` then walk ``model.graph.node`` in order, taking
constants from ``graph.initializer``. Map Conv (group == 1 -> conv2d,
group == C -> depthwise_conv2d), Gemm / MatMul+Add -> dense, BatchNormalization,
Relu, Clip(0, 6), MaxPool, AveragePool, GlobalAveragePool, Flatten / Reshape,
Add, Softmax, QuantizeLinear / DequantizeLinear. ONNX is NCHW with OIHW
filters: convert layouts as described in pytorch.py.
"""
from .base import Frontend, FrontendUnavailable, register_frontend


@register_frontend
class OnnxFrontend(Frontend):
    name = "onnx"
    extensions = (".onnx",)

    def accepts(self, source):
        if isinstance(source, str):
            return Frontend.accepts(self, source)
        return type(source).__module__.startswith("onnx")

    def load(self, source, **options):
        raise FrontendUnavailable(
            "the ONNX importer is not written yet: convert with onnx2tf to TFLite and pass "
            "the .tflite file (see ptensor/codegen/frontends/onnx.py)")
