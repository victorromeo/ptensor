"""Model importers. Each turns one framework's model into an ``ir.Graph``.

    from ptensor.codegen import frontends
    graph = frontends.load_model("model.tflite")            # picks the frontend
    graph = frontends.load_model(keras_model, framework="keras")

A new framework is one class::

    from ptensor.codegen.frontends import Frontend, register_frontend

    @register_frontend
    class MyFrontend(Frontend):
        name = "myfw"
        extensions = (".myfw",)
        def accepts(self, source):         # an object or path of this framework?
            ...
        def load(self, source, **options): # -> ir.Graph
            ...
"""
from .base import (Frontend, FrontendUnavailable, available_frontends,
                   get_frontend, load_model, register_frontend)
from . import keras as _keras            # noqa: F401  (registers itself)
from . import tflite as _tflite          # noqa: F401
from . import tf_savedmodel as _tf       # noqa: F401
from . import pytorch as _pytorch        # noqa: F401
from . import onnx as _onnx              # noqa: F401

__all__ = ["Frontend", "FrontendUnavailable", "available_frontends",
           "get_frontend", "load_model", "register_frontend"]
