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
         pytorch                quantize_int8
         onnx (stub)            plan_memory
```

* **`ir`** is a small framework-neutral graph: tensors (shape, dtype,
  optional constant data, quantisation, weight encoding) and nodes in
  execution order. It uses pTensor's layouts (NHWC, OHWI conv filters,
  `[1, kh, kw, C*M]` depthwise filters, `[in, out]` dense weights) so
  frontends do all the transposing and the emitter does none. Ops:
  `conv2d`, `depthwise_conv2d`, `dense`, `max_pool2d`, `avg_pool2d`,
  `global_avg_pool`, `batch_norm`, `activation` (31 kinds: relu, relu6,
  relu1, clip, sigmoid, tanh, leaky_relu, elu, celu, selu, gelu (erf and
  tanh forms), silu, mish, hard_sigmoid, hard_swish, softplus, softsign,
  log_sigmoid, abs, neg, square, sqrt, rsqrt, exp, log, floor, ceil, round,
  sin, cos), `add`, `binary` (sub, mul, div, maximum, minimum,
  squared_difference, pow, with broadcasting and constant operands),
  `reshape`, `softmax`, `pad`, `concat`, `transpose`, `mean`, `slice`,
  `matmul`, `lstm`, `gru`, `quantize`, `dequantize`.
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
  * `pytorch`: `nn.Module`s (exported with `torch.export` and an
    `example_input`), `ExportedProgram`s and `.pt2` files. See
    [PyTorch](#pytorch) below.
  * `onnx`: registered so ONNX files are recognised; `load()` raises
    `FrontendUnavailable` with the workaround (export to TFLite).
* **Passes** (`passes.py`) each return a new graph and add a line to
  `graph.notes`, which the generated file's header comment repeats.
* **Emitter** (`emitter.py`) writes the files. It includes only the
  operator and tensor headers the graph uses (not `pTensor.h`), checks with
  `#error` that pTensor was built with the features it needs, and writes a
  `.cmake` that forces those features ON and the unused ones OFF.
* **`reference`** runs float graphs in NumPy (weights decoded exactly as the
  device decodes them) and int8 graphs through pTensor's Python bindings;
  `gen.reference(x)` is the expected output of the generated code.

### PyTorch

```python
import torch
from ptensor.codegen import CodeGenerator

model = MyNet().eval()
gen = CodeGenerator(model, example_input=torch.randn(1, 3, 32, 32), name="mynet")
gen.generate("out/")
```

The importer exports the model with `torch.export` (in eval mode) and maps
the resulting ATen graph, so `nn` modules, `torch.nn.functional` calls and
tensor methods all arrive as the same operators. Anything that depends only
on constants (`torch.zeros`, weight reshapes, a positional embedding added
to a constant) is evaluated at import time.

**Layout.** PyTorch is NCHW and pTensor NHWC. Each value carries the
permutation between the two, so convolutions and pools run on NHWC data
without transposes, `permute` / `transpose` / `unsqueeze` cost nothing, and a
`transpose` node is emitted only where a PyTorch operator depends on the
element order (a `view` of a 4D tensor, a softmax over channels). A
`flatten` that feeds a `Linear` reorders the Linear's rows at generation
time instead of the data at run time. By default 4D inputs and outputs of
the generated function are NHWC; pass `channels_last_io=False` to keep
PyTorch's NCHW order there.

**Operators.** Every ATen operator pTensor has a kernel for, plus
compositions of those kernels for the rest:

| PyTorch | pTensor |
|---|---|
| `Conv1d`, `Conv2d` (any stride, dilation, `padding` int / tuple / `"same"` / `"valid"`, bias or not), depthwise (`groups == in_channels`, any multiplier) | conv2d / depthwise_conv2d; padding TF SAME cannot express becomes a `pad` node; other `groups` run as one dense conv with a block-diagonal filter |
| `Linear`, `F.linear`, `addmm`, `mm`, `matmul`, `bmm`, `@` | dense, or `matmul` for activation × activation and batched products (one 2D product per batch) |
| `BatchNorm1d/2d` | batch_norm (folded into the conv before it by `fuse`) |
| `LayerNorm`, `GroupNorm`, `InstanceNorm2d` | mean, sub, square, mean, add, rsqrt, mul (+ affine) |
| `MaxPool1d/2d`, `AvgPool1d/2d` (padding, `ceil_mode`, `count_include_pad`), `AdaptiveAvgPool1d/2d`, `AdaptiveMaxPool1d/2d` (output size dividing the input) | max_pool2d / avg_pool2d / global_avg_pool, with a `pad` node (−inf for max, 0 for average) when needed |
| `ReLU`, `ReLU6`, `Hardtanh`, `clamp`, `LeakyReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `GELU` (both), `SiLU`, `Mish`, `Hardsigmoid`, `Hardswish`, `Sigmoid`, `Tanh`, `Softplus` (any beta), `Softsign`, `LogSigmoid`, `Tanhshrink`, `Softmax`, `LogSoftmax`, `Softmin` (any dim) | activation, softmax |
| `+ - * /`, `rsub`, `reciprocal`, `pow`, `maximum`, `minimum`, `abs`, `neg`, `sqrt`, `rsqrt`, `exp`, `log`, `floor`, `ceil`, `round`, `sin`, `cos`, `square` | add / binary / activation, with PyTorch broadcasting and constant operands |
| `view`, `reshape`, `flatten`, `unflatten`, `squeeze`, `unsqueeze`, `permute`, `transpose`, `t`, `contiguous`, `cat`, `stack`, `split`, `chunk`, `unbind`, `narrow`, `select`, slicing with steps, `repeat`, `expand`, `F.pad` (constant) | free relabelling, reshape, transpose, slice, concat, pad |
| `mean`, `sum`, `var`, `std`, `amax` / `amin` / `max(dim)` / `min(dim)` over H, W | mean / global_avg_pool / max_pool2d |
| `nn.Upsample` / `F.interpolate` nearest with integer factors | concat + reshape |
| `LSTM`, `GRU` (stacked layers, `batch_first` or not, bias or not, zero initial state; output sequence and `h_n`) | lstm / gru: the pTensor cell, once per time step |
| `F.scaled_dot_product_attention` (mask, `is_causal`), `MultiheadAttention`, `TransformerEncoderLayer` | matmul, mul, add, softmax per head |
| `Dropout` and other identities in eval mode | removed |

Verification: `python/tests/test_pytorch.py` runs 128 operator and
small-model cases against PyTorch (NumPy reference and compiled C), and
[examples/codegen/verify_pytorch.py](../examples/codegen/verify_pytorch.py)
trains a CNN, a MobileNet-style net, a GRU and a small vision transformer
on MNIST and checks the generated C against PyTorch: every model agrees on
every test image's class, within 2.1e-5 in float
([PYTORCH_RESULTS.md](../examples/codegen/PYTORCH_RESULTS.md)).

`ptensor.codegen.frontends.pytorch.SUPPORTED` lists the ATen names. An
operator outside the table (transposed convolution, bilinear upsampling,
embedding, comparisons and `where`, bidirectional RNNs, reflect padding...)
raises `NotImplementedError` naming the operator and the graph node.

int8: `activations="int8"` works on PyTorch models whose operators have int8
kernels (conv, depthwise, dense, pooling, add, ReLU / ReLU6, sigmoid, tanh,
softmax, and the byte-moving pad, transpose, slice and reshape). The
calibration data is in the generated function's layout (NHWC by default).

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
generates 26 variants of an MNIST CNN, a small residual network
(depthwise, separable conv, batch norm, residual add) and a network with
average pooling, tanh and sigmoid, from Keras, TFLite and a SavedModel, compiles each, runs it on the test set and compares it with
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
* int8: average pooling, sigmoid and tanh are supported and TFLite-exact
  (sigmoid and softmax outputs use TFLite's fixed scale 1/256, zero point
  -128; tanh uses 1/128, 0; average pooling keeps its input's quantization).
  Global average pooling, standalone batch norm and standalone clip
  activations are not supported yet, nor are the float-only operator
  types (`binary`, `mean`, `concat`, `matmul`, `lstm`, `gru` and the
  activations beyond relu / relu6 / sigmoid / tanh); the generator says
  which node is the problem. Weight storage modes (packed, Hough,
  polynomial) apply to float activations.
* The reference float conv has no dilation; dilated convolutions need
  `fuse=True`.
* PyTorch: batch 1, static shapes, zero RNN initial state; the operators
  listed under [PyTorch](#pytorch). ONNX import is a placeholder (export to
  TFLite for now).
* `import ptensor` loads the C library, so the generator needs a
  `-DPTENSOR_BUILD_C_API=ON` build (or `PTENSOR_LIBRARY`) on the host.
