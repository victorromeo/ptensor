"""PyTorch (placeholder).

Registered so ``load_model(torch_module)`` is recognised and fails with a
clear message. Until this importer is written, export the model to TFLite
(ai-edge-torch: ``ai_edge_torch.convert(model, (example,)).export("m.tflite")``)
or to ONNX and use that frontend once it exists.

Writing it: trace with ``torch.fx.symbolic_trace(module)`` and map
nn.Conv2d / nn.Linear / nn.BatchNorm2d / nn.ReLU / nn.ReLU6 / nn.MaxPool2d /
nn.AvgPool2d / nn.AdaptiveAvgPool2d(1) / nn.Flatten / operator.add /
softmax to IR nodes. PyTorch is NCHW with OIHW filters, so transpose conv
filters to OHWI (``w.permute(0, 2, 3, 1)``), depthwise filters
``[C*M, 1, kh, kw]`` to ``[1, kh, kw, C*M]``, Linear weights ``[out, in]`` to
``[in, out]``, and permute the rows of the first Linear after a Flatten from
CHW to HWC order.
"""
from .base import Frontend, FrontendUnavailable, register_frontend


@register_frontend
class PyTorchFrontend(Frontend):
    name = "pytorch"
    extensions = (".pt", ".pth")

    def accepts(self, source):
        if isinstance(source, str):
            return Frontend.accepts(self, source)
        return any(c.__module__.startswith("torch.nn") for c in type(source).__mro__)

    def load(self, source, **options):
        raise FrontendUnavailable(
            "the PyTorch importer is not written yet: export the model to TFLite with "
            "ai-edge-torch and pass the .tflite file (see ptensor/codegen/frontends/pytorch.py)")
