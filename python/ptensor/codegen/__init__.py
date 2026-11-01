"""Generate pTensor C/C++ from a trained model.

    from ptensor.codegen import CodeGenerator
    gen = CodeGenerator("mnist_int8.tflite")                       # TFLite int8 as is
    gen = CodeGenerator(keras_model, weights="hough4", fuse=True)  # Keras, Hough weights
    gen = CodeGenerator(keras_model, activations="int8", calibration_data=x[:500])
    gen.generate("firmware/model", name="mnist")
    print(gen.report())

Layers:

    frontends/   one importer per framework -> ir.Graph (Keras, TFLite and
                 TF SavedModel now; PyTorch and ONNX registered as stubs)
    ir           the framework-neutral graph
    passes       prune, fuse / unfuse, encode_weights, quantize_int8, plan_memory
    emitter      ir.Graph -> <name>.h / <name>.cpp / <name>.cmake
    reference    NumPy interpreter for float graphs (tests, calibration)

Command line: ``python -m ptensor.codegen model.tflite -o out --name mnist``.
"""
from . import frontends, ir, passes, reference
from .emitter import CEmitter, EmitterError
from .frontends import (Frontend, FrontendUnavailable, available_frontends, load_model,
                        register_frontend)
from .generator import CodeGenerator, generate
from .ir import Graph

__all__ = ["CodeGenerator", "generate", "CEmitter", "EmitterError", "Frontend",
           "FrontendUnavailable", "Graph", "available_frontends", "frontends", "ir",
           "load_model", "passes", "reference", "register_frontend"]
