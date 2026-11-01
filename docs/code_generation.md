# Generating pTensor C++ from a model

`ptensor.codegen` turns a trained model into C++ source that runs it on
pTensor: the weights become `const` arrays (flash), the activations get one
statically planned arena, and every layer becomes one pTensor operator call.
The firmware calls two C functions:

```c
#include "mnist.h"
int8_t probs[MNIST_OUT0_SIZE];
mnist_invoke(image, probs);          /* image: int8_t[MNIST_IN0_SIZE] */
```

```python
from ptensor.codegen import CodeGenerator

gen = CodeGenerator("examples/mnist/mnist_int8.tflite", name="mnist")
gen.generate("firmware/model")       # mnist.h, mnist.cpp, mnist.cmake
print(gen.report())
```

```
model mnist (tflite), 8 nodes, ops: conv2d, dense, max_pool2d, reshape, softmax
options: activations=int8, fast_kernels=True, float_io=False, fuse=True, memory=planned, ...
weights (flash): 27424 B
activations: planned arena 7840 B (naive 13392 B, live peak 7840 B, 1 shared buffer(s))
pTensor features needed: PTENSOR_WITH_FAST_KERNELS, PTENSOR_WITH_QUANTIZED_OPS
```

The same from a shell: `python -m ptensor.codegen model.tflite -o out --name mnist`
(`--help` lists every flag).

## Options

Everything is a keyword argument of `CodeGenerator` (and a CLI flag).

| Option | Values | What it does |
| --- | --- | --- |
| **Quantisation** | | |
| `weights` | `float32` (default), `float16`, `int16`, `int8`, `uint2`..`uint7`, `hough2/3/4`, `quad2`..`quad16`, `cubic2`..`cubic16`, or a `ptensor.qat` WeightFormat | Storage of conv / depthwise / dense weights when activations are float: `PackedRomTensor`, `HoughRomTensor` or `PolyRomTensor`, decoded as the operator reads them. |
| `weight_encodings` | a `QATModel`, or `{layer name: Encoded}` | Use weights already encoded by quantisation-aware training instead of encoding them again, so the device runs exactly what QAT evaluated. |
| `activations` | `float32` (default), `int8` | `int8` is full-integer post-training quantisation, as the TFLite converter does it: int8 activations (range from calibration), symmetric per-channel int8 weights, int32 biases, run by the TFLM-compatible int8 kernels. A model that is already int8 (full-integer `.tflite`) is kept as it is. |
| `calibration_data` | float array, one example per row | Needed for `activations="int8"` on a float model. |
| `input_scale`, `input_zero_point` | | Fix the int8 input quantisation (e.g. `1/255`, `-128` for 0..1 images). |
| `float_io` | bool | With int8: quantize the input and dequantize the output inside `invoke()`, so it takes and returns float. |
| **Acceleration** | | |
| `fuse` | bool (default True) | Fold batch norms into the weights and run conv / depthwise / dense / add with their activation as one fused operator (`ops/fused`). False runs every graph node as its plain reference operator. |
| `fast_kernels` | bool (default True) | Build pTensor with the dense pointer fast paths (`PTENSOR_WITH_FAST_KERNELS`). |
| `weights_in_ram` | bool | Decode packed / Hough / polynomial weights to float in RAM once in `init()` (float speed, costs RAM) instead of decoding each read from flash. |
| **Pruning** | | |
| `prune` | fraction, or `{layer: fraction}` | Structured pruning: remove the lowest-L1 filters / units of every layer whose channels feed exactly one later conv / dense, and the matching input channels there. The result is a smaller dense network. One-shot, so it costs accuracy; for trained pruning use `ptensor.prune.PrunedModel` and generate from its pruned model. |
| `prune_criterion`, `prune_multiple_of` | `l1` / `l2`, int | |
| **Memory** | | |
| `memory` | `planned` (default), `dynamic` | `planned`: one static arena, offsets computed by the generator (greedy by size, reshape outputs as views, elementwise ops in place when their input is dead). `dynamic`: `RamTensor`s from pTensor's RAM allocator, freed after their last use. |
| `alignment` | bytes (default 8) | Arena alignment. |
| `meta_arena_bytes`, `ram_arena_bytes` | | Override the allocator sizes the generator picks (also overridable with `-D<NAME>_META_ARENA_BYTES=...`). |

The options combine: `CodeGenerator(model, prune=0.25, activations="int8",
calibration_data=x)` prunes, folds, quantises and plans in that order.

## How it is built

```
model ──frontend──▶ ir.Graph ──passes──▶ ir.Graph ──emitter──▶ name.h / name.cpp / name.cmake
         keras                  prune
         tflite                 fuse / unfuse
         tf (SavedModel)        encode_weights
         pytorch (stub)         quantize_int8
         onnx (stub)            plan_memory
```

* **`ir`** is a small framework-neutral graph: tensors (shape, dtype,
  optional constant data, quantisation, weight encoding) and nodes in
  execution order. It uses pTensor's layouts (NHWC, OHWI conv filters,
  `[1, kh, kw, C*M]` depthwise filters, `[in, out]` dense weights) so
  frontends do all the transposing and the emitter does none. Ops:
  `conv2d`, `depthwise_conv2d`, `dense`, `max_pool2d`, `avg_pool2d`,
  `global_avg_pool`, `batch_norm`, `activation` (relu, relu6, relu1, clip,
  sigmoid, tanh), `add`, `reshape`, `softmax`, `quantize`, `dequantize`.
* **Frontends** (`ptensor/codegen/frontends/`) each implement
  `accepts(source)` and `load(source) -> Graph` and register themselves;
  `load_model()` picks the first that accepts the source.
  * `keras`: Sequential and functional Keras 3 / tf.keras models or
    `.keras` / `.h5` files (Conv2D, DepthwiseConv2D, SeparableConv2D, Dense,
    BatchNormalization, ReLU, Activation, Softmax, Max/AveragePooling2D,
    GlobalAveragePooling2D, Flatten, Reshape, Add, Dropout).
  * `tflite`: float32 and full-integer int8 flatbuffers, read with the
    `tflite` package (no TensorFlow needed). Shape arithmetic that Keras
    exports around `RESHAPE` is dropped (shapes are static).
  * `tf`: SavedModel directories and concrete functions, converted to a
    float TFLite flatbuffer with TensorFlow's own converter and then
    imported as TFLite.
  * `pytorch`, `onnx`: registered so their models are recognised; `load()`
    raises `FrontendUnavailable` with the workaround (export to TFLite) and
    the module docstrings describe the layer mapping to implement.
* **Passes** (`passes.py`) each return a new graph and add a line to
  `graph.notes`, which the generated file's header comment repeats.
* **Emitter** (`emitter.py`) writes the files. It includes only the
  operator and tensor headers the graph uses (not `pTensor.h`), checks with
  `#error` that pTensor was built with the features it needs, and writes a
  `.cmake` that forces those features ON and the unused ones OFF.
* **`reference`** runs float graphs in NumPy (weights decoded exactly as the
  device decodes them) and int8 graphs through pTensor's Python bindings;
  `gen.reference(x)` is the expected output of the generated code.

### Adding a framework

```python
from ptensor.codegen import Frontend, register_frontend
from ptensor.codegen.ir import Graph

@register_frontend
class MyFrontend(Frontend):
    name = "myfw"
    extensions = (".myfw",)

    def accepts(self, source):
        return isinstance(source, MyModel) or Frontend.accepts(self, source)

    def load(self, source, **options):
        g = Graph(source.name)
        x = g.add_tensor("input", (1, 28, 28, 1)); g.inputs.append(x.name)
        ...  # g.add_const(...), g.add_node("conv2d", [x, w, b], [y], {...})
        return g
```

Everything after the frontend (passes, emitter, every flag) then works for
the new framework unchanged.

## The generated code

`mnist.cpp` (int8 TFLite model, abridged):

```cpp
#include "pTensor/ops/tflm/convolution2d.hpp"     // only what this model uses
#include "pTensor/ops/tflm/fully_connected.hpp"
...
#if !PTENSOR_WITH_QUANTIZED_OPS
#error "mnist needs pTensor built with PTENSOR_WITH_QUANTIZED_OPS=ON (see mnist.cmake)"
#endif

alignas(8) const int8_t k0_conv2d[72] MNIST_WEIGHT_ATTR = { ... };
alignas(8) uint8_t g_arena[7840];  // planned arena 7840 B (naive 13392 B, ...)

extern "C" int mnist_invoke(const int8_t* in0, int8_t* out0) {
  ...
  // 0: conv_2d_0 (conv2d, relu) -> [1, 28, 28, 8]
  t1 = new BufferTensor(TensorShape(1, 28, 28, 8), i8, g_arena + 0);
  t1->set_quantization_params(PerTensorQuantizationParams(-128, 0.0104707247f));
  {
    Tensor w = new RomTensor(TensorShape(8, 3, 3, 1), i8, k0_conv2d);
    ...
    Op op(strides, SAME, dilation, static_cast<TFLM::TfLiteFusedActivation>(1));
    op.set_inputs({{Op::in, t0}, {Op::filter, w}, {Op::bias, b}}).set_outputs({{Op::out, t1}}).eval();
  }
  if (g_err.failed) return -2;
  t0.free();
  ...
}
```

Hooks: define `<NAME>_WEIGHT_ATTR` to place the weights in a section,
`<NAME>_META_ARENA_BYTES` / `<NAME>_RAM_ARENA_BYTES` to resize the
allocators (`<name>_meta_high_water()` / `<name>_ram_high_water()` report
what was used), and `<NAME>_EXTERNAL_CONTEXT` to keep your own allocators on
pTensor's default context.

### Building it into firmware

```cmake
include(model/mnist.cmake)              # before pTensor: sets PTENSOR_WITH_* for this model
add_subdirectory(third_party/ptensor ptensor)
add_executable(app main.c)
target_link_libraries(app mnist)
```

The `.cmake` forces the features the model needs ON and sets the others OFF
unless the project already set them, so several models can share one
pTensor build. For the int8 MNIST model only `QUANTIZED_OPS` and
`FAST_KERNELS` stay on; Hough, polynomial, fusion and SimpleErrorHandler are
not compiled.

## Verification

[examples/codegen/verify_models.py](../examples/codegen/verify_models.py)
generates 24 variants of an MNIST CNN and a small residual network
(depthwise, separable conv, batch norm, residual add) from Keras, TFLite and
a SavedModel, compiles each, runs it on the test set and compares it with
Python. Results are in
[examples/codegen/RESULTS.md](../examples/codegen/RESULTS.md). In short:

* every float mode matches the NumPy reference of its graph within 3.3e-6
  and, with float32 weights, the original Keras / TFLite / SavedModel output
  within 2e-6;
* every int8 mode is bit-exact with pTensor's kernels through the Python
  bindings, and the int8 `.tflite` models are bit-exact with TFLite's
  reference kernels up to the softmax (the softmax is within 5/256, from
  TFLite's fixed-point exp);
* modes built through their generated `.cmake` (unused features OFF) give
  identical output to the full-feature build.

Writing the verifier found a bug in pTensor's int8 softmax
(`sq_softmax_k`): it subtracted the raw int8 maximum from dequantized
values, so on a confident prediction every exponent underflowed and every
class came out as 127. It now takes the maximum over dequantized values and
rounds to nearest; `tests/operators/test_sq_softmax.cpp` has the regression
tests.

## Limits

* Batch 1, single subgraph, at most 4 dimensions of at most 65535 each
  (pTensor's `TensorShape`).
* int8: no TFLite-exact pTensor kernel yet for average pooling, global
  average pooling, sigmoid, tanh or standalone batch norm; the generator
  says which node is the problem. Weight storage modes (packed, Hough,
  polynomial) apply to float activations.
* The reference float conv has no dilation; dilated convolutions need
  `fuse=True`.
* PyTorch and ONNX import are placeholders (export to TFLite for now).
* `import ptensor` loads the C library, so the generator needs a
  `-DPTENSOR_BUILD_C_API=ON` build (or `PTENSOR_LIBRARY`) on the host.
