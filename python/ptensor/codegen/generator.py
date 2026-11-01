"""CodeGenerator: model in, pTensor C/C++ out."""
import os

import numpy as np

from . import passes, reference
from .emitter import CEmitter, c_ident
from .frontends import load_model


class CodeGenerator(object):
    """Generate pTensor C/C++ for a model.

        gen = CodeGenerator(keras_model, weights="uint4", fuse=True, memory="planned")
        gen.generate("out/", name="mnist")     # mnist.h, mnist.cpp, mnist.cmake
        print(gen.report())

    model: a Keras model, a .tflite / .keras / .h5 path or flatbuffer bytes,
    a TensorFlow SavedModel directory, an ``ir.Graph``, or anything a
    registered frontend accepts (see ``ptensor.codegen.frontends``).
    framework: force a frontend ("keras", "tflite", "tf", "pytorch", "onnx").
    example_input: for a PyTorch nn.Module, an example input (tensor or
    array, or a tuple of them) used to export it with torch.export.
    channels_last_io: PyTorch only; True (default) makes 4D inputs and outputs
    NHWC in the generated function, False keeps PyTorch's NCHW there.

    Quantisation
      weights      storage of conv / dense weights when activations are float:
                   any ``ptensor.qat.MODES`` name ("float32", "float16",
                   "int16", "int8", "uint2".."uint7", "hough2/3/4",
                   "quad2".."quad16", "cubic2".."cubic16") or a WeightFormat.
      weight_encodings  weights encoded elsewhere, e.g. a QATModel (its
                   export() is used) or {layer name: Encoded}; these win over
                   ``weights`` for the layers they name.
      activations  "float32", or "int8" for full-integer post-training
                   quantisation (int8 activations and per-channel weights,
                   int32 biases, TFLM kernels). Needs calibration_data unless
                   the model is already int8 (a full-integer .tflite).
      calibration_data  float examples for "int8" (an array, one per row).
      input_scale, input_zero_point  fix the int8 input quantisation.
      float_io     with "int8": quantize the input and dequantize the output
                   inside the generated function so it takes and returns float.

    Acceleration
      fuse         fold batch norms into weights and run conv / depthwise /
                   dense / add with their activation in one fused operator.
                   False runs every node as the plain reference operator.
      fast_kernels build pTensor with the dense pointer fast paths.
      weights_in_ram  decode packed / Hough / polynomial weights to float in
                   RAM once at init (fast, costs RAM) instead of decoding each
                   read from flash (small, slower).

    Pruning
      prune        fraction of filters / units to remove from each layer
                   (structured, lowest L1 norm first), or {layer: fraction}.
                   One-shot pruning costs accuracy; for trained pruning use
                   ptensor.prune.PrunedModel and pass the pruned model here.
      prune_criterion, prune_multiple_of

    Memory
      memory       "planned": one static arena with offsets computed now
                   (greedy by size, in-place reuse); "dynamic": RamTensors
                   from pTensor's RAM allocator, freed after their last use.
      alignment    arena alignment in bytes.
      meta_arena_bytes, ram_arena_bytes  override the allocator sizes.
    """

    def __init__(self, model, framework=None, name=None, weights="float32",
                 weight_encodings=None, activations="float32", calibration_data=None,
                 input_scale=None, input_zero_point=None, float_io=False, fuse=True,
                 fast_kernels=True, weights_in_ram=False, prune=0.0, prune_criterion="l1",
                 prune_multiple_of=1, memory="planned", alignment=8, meta_arena_bytes=None,
                 ram_arena_bytes=None, encode_min_size=64, example_input=None,
                 channels_last_io=True):
        if activations not in ("float32", "int8"):
            raise ValueError("activations must be 'float32' or 'int8'")
        if memory not in ("planned", "dynamic"):
            raise ValueError("memory must be 'planned' or 'dynamic'")
        self.options = dict(weights=getattr(weights, "name", weights), activations=activations,
                            fuse=fuse, fast_kernels=fast_kernels, weights_in_ram=weights_in_ram,
                            prune=prune, memory=memory, float_io=float_io)
        frontend_options = {}
        if example_input is not None:
            frontend_options["example_input"] = example_input
        if not channels_last_io:
            frontend_options["channels_last_io"] = False
        self.source_graph = load_model(model, framework, **frontend_options)
        self.name = c_ident(name or self.source_graph.name or "model")
        g = self.source_graph.copy()
        source_int8 = g.is_quantized()

        if prune:
            if source_int8:
                raise ValueError("cannot prune an already quantised model")
            g = passes.prune(g, prune, prune_criterion, prune_multiple_of)

        if source_int8 or activations == "int8":
            if weights not in (None, "float32") or weight_encodings:
                raise ValueError("weight storage modes apply to float activations; int8 models "
                                 "store int8 weights")
            g = passes.fuse(g)
            if not source_int8:
                if calibration_data is None:
                    raise ValueError("activations='int8' needs calibration_data")
                g = passes.quantize_int8(g, calibration_data, input_scale, input_zero_point,
                                         float_io)
            self.options["activations"] = "int8"
            fuse = True  # int8 kernels always fuse their activation
        else:
            g = passes.fuse(g) if fuse else passes.unfuse(g)
            if hasattr(weight_encodings, "export"):
                weight_encodings = {layer.name: enc for layer, enc, _ in weight_encodings.export()}
            if weights not in (None, "float32") or weight_encodings:
                g = passes.encode_weights(g, None if weights in (None, "float32") else weights,
                                          overrides=weight_encodings, min_size=encode_min_size)
        self.graph = g.validate()
        self.plan = passes.plan_memory(g, alignment) if memory == "planned" else None
        if self.plan is not None:
            g.notes.append("plan_memory: " + self.plan.describe())
        self._emitter_args = dict(fused=fuse, weights_in_ram=weights_in_ram,
                                  fast_kernels=fast_kernels, meta_arena_bytes=meta_arena_bytes,
                                  ram_arena_bytes=ram_arena_bytes)
        self.files = None

    # -- outputs --------------------------------------------------------------

    def emit(self):
        """{file name: text} without writing anything."""
        opts = ", ".join("%s=%s" % kv for kv in sorted(self.options.items()))
        em = CEmitter(self.graph, self.name, self.plan, options_comment=opts, **self._emitter_args)
        self.files = em.emit()
        self.features = sorted(em.features)
        self.weight_bytes = em.flash_bytes
        self.ram_weight_bytes = em.ram_weight_bytes
        return self.files

    def generate(self, out_dir, name=None):
        """Write <name>.h, <name>.cpp and <name>.cmake into out_dir. Returns
        the list of paths."""
        if name:
            self.name = c_ident(name)
        files = self.emit()
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for fname, text in files.items():
            path = os.path.join(out_dir, fname)
            with open(path, "w") as f:
                f.write(text)
            paths.append(path)
        return paths

    def reference(self, x):
        """Expected output of the generated code for one float input, from
        the NumPy interpreter on the processed graph (float graphs only;
        weights decoded exactly as the device decodes them)."""
        return reference.run(self.graph, x)

    def report(self):
        if self.files is None:
            self.emit()
        g = self.graph
        acts = g.activations()
        lines = ["model %s (%s), %d nodes, ops: %s" % (self.name, self.source_graph.source,
                                                       len(g.nodes), ", ".join(g.ops_used())),
                 "options: " + ", ".join("%s=%s" % kv for kv in sorted(self.options.items())),
                 "weights (flash): %d B" % self.weight_bytes]
        if self.ram_weight_bytes:
            lines.append("weights decoded to RAM: %d B" % self.ram_weight_bytes)
        if self.plan is not None:
            lines.append("activations: " + self.plan.describe())
        else:
            lines.append("activations: dynamic RamTensors (%d B if all were live)"
                         % sum(g.tensors[n].nbytes for n in acts))
        lines.append("pTensor features needed: " + (", ".join(self.features) or "none"))
        lines += ["pass: " + n for n in g.notes]
        return "\n".join(lines)


def generate(model, out_dir, name=None, **options):
    """One call: CodeGenerator(model, name=name, **options).generate(out_dir)."""
    gen = CodeGenerator(model, name=name, **options)
    gen.generate(out_dir)
    return gen


__all__ = ["CodeGenerator", "generate"]
