# Operator fusion

A conv, depthwise conv or dense layer is usually followed by a batch norm and
an activation, and a residual add by a ReLU. Run as separate operators, each
step reads and writes the whole feature map, and each step's output is a
tensor that needs RAM. Fusion does the whole chain in one pass, while the
value is still in a register.

Everything here is optional at compile time: `-DPTENSOR_WITH_FUSION=ON` (the
default) makes the fused operators available through `pTensor.h`, the C API
and Python. The operators are header-only templates, so a firmware that does
not instantiate one is byte-for-byte unchanged (the Cortex-M4 size probe is
the same size either way).

## What is fused

| Pattern | Operator | Notes |
| --- | --- | --- |
| conv2d → [batch norm] → act | `FusedConv2dOperator<float>` | OHWI filter, stride, dilation, SAME/VALID |
| depthwise → [batch norm] → act | `FusedDepthwiseConv2dOperator<float>` | depth multiplier, dilation |
| dense → [batch norm] → act | `FusedFullyConnectedOperator<float>` | input can be a 4D feature map (no reshape) |
| add → act | `FusedAddOperator<float>` | NumPy broadcasting |
| int8 add → act → requantise | `TflmSymQuantOps::QuantizedAddOperator<int8_t>` | bit-exact with TFLite `ADD` |

The int8 conv, depthwise conv and fully connected operators already fused
their activation and requantisation (the TFLM kernels clamp the requantised
accumulator to the activation range), so the int8 gap was ADD; residual
networks need it.

The activation is a clamp, `FusedActivation{lo, hi}`: `none()`, `relu()`,
`relu6()`, `relu1()` or `clamp(lo, hi)`. Those are all the activations TFLite
fuses into a layer. Consecutive clamps compose (`relu` then `relu6` is
`relu6`), which `ptensor.fusion.fuse` uses.

Each conv / dense operator takes inputs `in`, `filter`, `bias` (optional) and
`scale` (optional) and computes, per output channel `c`,

```
out = act(acc * scale[c] + bias[c])      if scale is given
out = act(acc + bias[c])                 otherwise
```

## Folding a batch norm

At inference a batch norm is a per-channel affine map, so

```
bn(layer(x) + b) = layer(x) * s + ((b - mean) * s + beta)
s = gamma / sqrt(variance + epsilon)
```

There are two ways to use it:

1. **Fold into the weights** (recommended). Multiply `s` into the filter
   (`fusion::fold_conv2d_weights`, `fold_depthwise_weights`,
   `fold_dense_weights` in C++, `ptensor.fusion.fold_conv2d` etc. in Python)
   and pass only the new bias. The batch norm then costs nothing at run time.
   Do this on the host before exporting the weights, so quantisation and
   packing see the folded values.
2. **Pass `scale`**. The weights stay as they are and the operator applies
   `s` in its epilogue (one multiply per output). Use this when the weights
   cannot be rewritten, e.g. they are already packed in flash as a
   `PackedRomTensor` / `HoughRomTensor` / `PolyRomTensor`.

```cpp
#include "pTensor.h"
using namespace pTensor;
using namespace pTensor::ReferenceOperators;

float scale[OC], bias[OC];
fusion::fold_batch_norm(mean, var, gamma, beta, 1e-3f, conv_bias, OC, scale, bias);
fusion::fold_conv2d_weights(filter, OC, KH * KW * IC, scale);   // option 1

typedef FusedConv2dOperator<float> Op;
Op op(1, 1, SAME, FusedActivation::relu6());
op.set_inputs({{Op::in, x}, {Op::filter, w}, {Op::bias, b}})
  .set_outputs({{Op::out, y}})
  .eval();
```

## Python

`ptensor.fusion` has the folding helpers, the fused kernels, and a pass over a
simple sequential layer list:

```python
from ptensor import fusion

layers = fusion.from_keras(model)          # Conv2D, DepthwiseConv2D, Dense,
                                           # BatchNormalization, ReLU, pooling...
fused, report = fusion.fuse(layers)        # fold_weights=False keeps weights as is
y = fusion.run(fused, x)                   # runs on pTensor's kernels
```

`fuse` folds a batch norm only when it directly follows the layer (a batch
norm after a ReLU cannot be folded and is left in place), then absorbs the
activations that follow.

## Correctness

- Activation-only fusion adds the terms in the same order as the unfused
  operators, so the result is **bit-identical** to running the layer and then
  the activation (checked for conv, depthwise, dense and add).
- Batch-norm folding changes the rounding, not the maths: the fused output is
  within 4e-6 of `layer → BatchNormOperator → activation` in the benchmark
  (largest difference on any output; values are O(1)).
- The int8 add is **bit-exact with TFLite**: the expected outputs in
  `tests/operators/constants_int8_add.hpp` come from TFLite 2.21 itself
  (BUILTIN_REF kernels, `tests/operators/gen_constants_int8_add.py`), and
  `python/tests/test_fusion.py` compares against TFLite directly when
  TensorFlow is installed.
- A Keras Conv2D → BN → ReLU6 → DepthwiseConv2D → BN → ReLU → Conv2D → BN →
  MaxPool → Dense model gives the same output as TensorFlow (to 1e-4) both
  unfused and fused (`test_keras_conv_bn_relu_model`).
- Non-dense tensors (packed weights, RAM tensors) take a fallback path with
  the same arithmetic; tests check it matches the dense path bit for bit.

## Results

Host benchmark: `-DPTENSOR_BENCH=ON`, then
`build/examples/bench/ptensor_fusion_bench` (x86-64 Xeon 2.1 GHz, Release,
best of N runs, microseconds).

- **unfused**: the pTensor operators one after another.
- **unfused\***: the layer op, then the batch norm and activation as tight
  dense loops over a precomputed scale and shift. This is the best a
  non-fused pipeline could do, so the gap from here to *fused* is the gain
  from fusion alone.
- **fused**: fused operator with `scale`. **folded**: batch norm folded into
  the weights.

| Pattern | unfused | unfused\* | fused | folded | speed-up vs unfused | intermediate RAM removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conv2d 1x32x32x16, 3x3 → 32, + BN + ReLU6 | 3540 | 2032 | 2061 | 2015 | 1.76x | 256 KB |
| depthwise 1x32x32x64, 3x3, + BN + ReLU6 | 3083 | 129 | 133 | 132 | 23x | 512 KB |
| dense 512 → 256, + BN + ReLU | 38.9 | 22.3 | 22.5 | 18.2 | 2.1x | 2 KB |
| add 1x32x32x64 + ReLU | 31.2 | 21.3 | 21.3 | – | 1.46x | 256 KB |
| int8 add 1x32x32x64 + ReLU | 2238 | – | 1107 | – | 2.0x | 64 KB |
| MobileNetV2-style block (expand, depthwise, project, residual) | 4305 | – | – | 544 | 7.9x | 0.96 MB → 0.33 MB |

Dense with ReLU only (no batch norm): 22.1 µs with `FullyConnectedOperator`
(its activation is a `std::function` call per output) vs 18.3 µs fused, with
identical output.

What the numbers say:

- Most of the speed-up against *unfused* comes from not running
  `BatchNormOperator`, which reads every element through a virtual call and
  takes a square root per element. Folding removes it entirely.
- Against the best-case unfused pipeline (*unfused\**) the host gain is small
  (0–20%): on a desktop the extra passes stream through cache quickly. The
  bigger win on a microcontroller is memory: each fused step removes one or
  two feature-map-sized intermediate tensors (the block above needs a third of
  the activation RAM), and there is no cache to hide the extra passes.
- The int8 baseline is the int8 add followed by pTensor's standalone int8
  `ReLUOperator`. That operator used to dequantise to float and truncate when
  requantising (2283 of 65536 outputs one step off TFLite); it now uses
  TFLite's fixed-point arithmetic, so fused and unfused agree exactly.

Cortex-M4 (`-Os`, newlib-nano), a conv 8x8x4 → 8 + BN + ReLU6 program:

| | flash | RAM (.bss) |
| --- | ---: | ---: |
| Conv2d → BatchNorm → Clip | 11,712 B | 12,468 B |
| FusedConv2d (scale input) | 9,560 B | 8,372 B |

The fused version is 2.1 KB smaller (one operator instead of three) and
needs 4 KB less RAM (the two intermediate 8x8x8 float tensors). Cycle counts
on hardware were not measured.
