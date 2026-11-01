# Operator audit

Every operator in `src/pTensor/ops` was checked against a trusted reference:
TFLite's reference kernels (`OpResolverType.BUILTIN_REF`) for the int8
operators, and NumPy or a double-precision loop with TensorFlow semantics
for the float ones. Each operator with a pointer fast path was also run with
tensors that hide their raw pointer, so the per-element reference path (the
one PackedRomTensor, HoughRomTensor and PolyRomTensor weights take, and the
only one when `PTENSOR_WITH_FAST_KERNELS=OFF`) is checked too.

The audit found 16 bugs across 17 operators. All are fixed, and each fix has a
test that fails on the old code.

## Where the checks live

| Test | Reference | Covers |
|---|---|---|
| `tests/operators/test_op_audit.cpp` | double-precision loops with TensorFlow semantics, in the test | the reference operators the C API does not reach, on both paths |
| `tests/operators/test_op_audit_tflite.cpp` | 18 single-operator TFLite models (`constants_op_audit_tflite.hpp`, made by `gen_constants_op_audit_tflite.py`) | int8 conv, depthwise conv, max / average pool, logistic, tanh, softmax: all bit-exact |
| `python/tests/test_ptensor.py`, `test_layers.py`, `test_fusion.py` | NumPy, and TFLite-exact integer arithmetic for int8 | everything the C API exposes; new SAME-pooling and large-stride sweeps |
| `examples/codegen/verify_models.py` | NumPy interpreter, Keras, TFLite | whole generated models, float and int8 |

To regenerate the TFLite constants (needs TensorFlow):

    python tests/operators/gen_constants_op_audit_tflite.py tests/operators/constants_op_audit_tflite.hpp

## Status per operator

"Fixed" means the operator was wrong before this audit and is right now.

### Float and generic reference operators (`ops/reference`)

| Operator | Types | Status | Finding |
|---|---|---|---|
| `AddOperator` | any | OK | |
| `Sub` / `Mul` / `Div` / `Maximum` / `Minimum` / `SquaredDifference` / `Pow` / `BroadcastAdd` | any | OK | broadcasting checked on both paths |
| 22 unary activations (`LeakyReLU` ... `Clip`) | float | OK | |
| `ReLUOperator`, `ReLU6Operator`, in-place forms | float | OK | |
| `LogisticOperator` | float | OK | |
| `InPlaceLogistic` | float | **Fixed** | ran softmax instead of sigmoid |
| `TanhOperator` | float | OK | |
| `SoftmaxOperator`, `InPlaceSoftmax` | float | **Fixed** | rank 3 and 4 inputs were indexed as 2D, reading and writing out of bounds (heap corruption); now any rank, over the last axis |
| `Conv2dOperator` | float | **Fixed** | padding offset went negative when the stride was larger than the kernel (SAME with k < s, VALID with (in - k) mod s >= 3), shifting every window; both paths |
| `DepthwiseSeparableConvOperator` (depthwise then pointwise) | float | **Fixed** | same padding offset as `Conv2dOperator` |
| `DepthwiseConv2dOperator` | float | OK | dilation and depth multiplier checked on both paths |
| `MatrixMultOperator`, `MatrixMultOperatorV2`, `FullyConnectedOperator` | float | OK | |
| `MaxPoolOperator`, `MinPoolOperator` | float, int8 | **Fixed** | SAME padding fed 0 into the window, so a max over negative values returned 0 (and a min over positive ones); padding offset as in conv |
| `AvgPoolOperator` | float | **Fixed** | SAME divided by the full window instead of the taps inside the image |
| `GlobalAveragePoolOperator` | float | OK | |
| `BatchNormOperator`, `ChannelAffineOperator`, `PadOperator`, `ConcatOperator` | float | OK | |
| `LstmCellOperator`, `GruCellOperator` | float | OK | both paths |
| `ArgMaxOperator`, `ArgMinOperator` | any | **Fixed** | ties returned the last index (NumPy and TFLite return the first); an out-of-range axis now raises an error |
| `MaxOperator`, `MinOperator` (whole tensor) | any | OK | |
| `ReduceMeanOperator` | any | **Fixed** | did not compile if used (private base, wrong constructor type, uninitialised axes, non-inline definitions in a header); rewritten |
| `TransposeOperator` | any | **Fixed** | applied the inverse permutation, so any permutation that is not its own inverse, such as (1, 2, 0), gave the wrong layout and shape; also raised InvalidResizeError for every BufferTensor output |
| `ReshapeOperator` | any | **Fixed** | raised InvalidResizeError for every BufferTensor output, even of the right shape |
| `SqueezeOperator` | any | **Fixed** | stored dimensions in the element type (an int8 tensor lost any dimension over 127) and never updated the rank |

### int8 operators

| Operator | Status | Finding |
|---|---|---|
| `TflmSymQuantOps::Conv2dOperator` | **Fixed** | dilation was accepted but ignored (and code generation emits dilated int8 convs); padding offset as in float conv. Now bit-exact with TFLite for 3x3, dilated, 1x1 and 2x2 stride 4, and 2x1 strides |
| `TflmSymQuantOps::DepthwiseSeparableConvOperator` | **Fixed** | SAME padding ignored dilation; stride height and width were swapped; per-tensor weights raised an error (now broadcast, as TFLite does). Bit-exact for dilated, multiplier 2 and 1x2-stride cases |
| `TflmSymQuantOps::FullyConnectedOperator` (per-tensor, per-channel) | OK | bit-exact (Python tests, code generation against TFLite) |
| `TflmSymQuantOps::QuantizeOperator`, `DequantizeOperator` | OK | |
| `TflmSymQuantOps::QuantizedAddOperator` | OK | bit-exact (tests from the fusion work) |
| `ReLUOperator<int8_t>`, `ReLU6Operator<int8_t>` | OK | bit-exact on all 256 inputs |
| `LogisticOperator<int8_t>` | **Fixed** | truncated instead of rounding: 127 of 256 inputs one step off TFLite; now bit-exact on all 256 (and used by code generation) |
| `TanhOperator<int8_t>` | OK | bit-exact on all 256 inputs |
| `SoftmaxOperator<int8_t>` | **Fixed** | rank 3 and 4 inputs, as for float; bit-exact on 2D and 4D cases |
| `MaxPoolOperator<int8_t>` | **Fixed** | SAME padding, as for float; now bit-exact |
| `AvgPoolOperator<int8_t>` | **Fixed** | summed in int8 (overflowed), truncated, and counted padding; now accumulates in int32, rounds half away from zero, and is bit-exact with TFLite |
| `Conv2dOperator<int8_t>` (float-emulated, reference ops) | **Fixed** | truncated instead of rounding to nearest |

### Fused operators (`ops/fused`)

| Operator | Status |
|---|---|
| `FusedConv2dOperator`, `FusedDepthwiseConv2dOperator`, `FusedFullyConnectedOperator`, `FusedAddOperator` | OK: already used TensorFlow padding; bit-identical to the unfused chain on dense and packed weights |

## Known limits (not bugs, left as they are)

* `InPlaceReLU<int8_t>` / `InPlaceReLU6<int8_t>` clamp raw integers at 0 and
  6, ignoring quantization. Use `ReLUOperator<int8_t>` for quantized tensors.
* `GlobalAveragePoolOperator` and `ReduceMeanOperator` on integer types
  average raw values with one rounding; TFLite's int8 MEAN requantizes, so
  they are not TFLite-exact (code generation already rejects int8 global
  average pooling).
* The reference `Conv2dOperator<float>` has no dilation (code generation
  uses the fused operator for dilated float convs).
