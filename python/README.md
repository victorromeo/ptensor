# pTensor for Python

Python bindings that call the same C++ kernels pTensor runs on the device, so
you can check a model layer by layer against NumPy or TensorFlow without
writing C.

## Build and use

```sh
# from the repository root
cmake -S . -B build -DPTENSOR_BUILD_C_API=ON
cmake --build build
pip install -e python              # or: export PYTHONPATH=$PWD/python
python3 -c "import ptensor; print(ptensor.__version__)"
```

The package finds `libptensor_c` in this order: the `PTENSOR_LIBRARY`
environment variable, next to the package, then `build*/src/pTensor/` in the
repository checkout.

```python
import numpy as np
import ptensor

x = np.random.rand(1, 28, 28, 1).astype(np.float32)
w = np.random.rand(8, 3, 3, 1).astype(np.float32)   # OHWI
y = ptensor.maxpool2d(ptensor.relu(ptensor.conv2d(x, w, padding="SAME")), 2)

# Pack weights as 6-bit codes for a PackedRomTensor and see what ops will read
codes = np.random.randint(0, 64, 100)
packed = ptensor.pack_bits(codes, 6)             # 75 bytes
values = ptensor.decode_packed(packed, ptensor.PackedFormat.UINT_BITS, 100,
                               bits=6, scale=0.02, zero_point=32)
```

Available:

- Dense and conv: `matmul`, `fully_connected`, `conv2d`, `depthwise_conv2d`
- Pooling: `maxpool2d`, `avgpool2d`, `global_avgpool2d`
- Recurrent: `lstm_cell`, `gru_cell` (Keras weight layout)
- Normalisation and shape: `batch_norm`, `softmax`, `pad2d`, `concat`
- Elementwise with broadcasting: `add` (equal shapes), `broadcast_add`, `sub`,
  `mul`, `div`, `maximum`, `minimum`, `squared_difference`, `power`, or
  `binary(name, a, b)`
- Activations and math: `relu`, `relu6`, `sigmoid`, `tanh`, `leaky_relu`,
  `elu`, `clip`, and `unary(name, x)` for everything in `ptensor.UNARY_OPS`
  (selu, hard_sigmoid, hard_swish, swish, gelu, softplus, softsign, abs, neg,
  square, sqrt, rsqrt, exp, log, floor, ceil, round, sin, cos)
- Quantisation (TFLite int8 arithmetic): `quantize`, `dequantize`,
  `fully_connected_int8` (per-tensor or per-channel weights), `conv2d_int8`,
  `maxpool2d_int8`, `softmax_int8`
- Packing: `pack_bits`, `decode_packed`, `packed_size`, `float_to_half`,
  `half_to_float`

- Fusion (`ptensor.fusion`): `fold_batch_norm`, `fold_conv2d`,
  `fold_depthwise_conv2d`, `fold_dense`; fused kernels `fused_conv2d`,
  `fused_depthwise_conv2d`, `fused_fully_connected`, `fused_add`, `add_int8`
  (TFLite-exact); and `from_keras(model)` → `fuse(layers)` → `run(layers, x)`
  to fold and fuse a Keras Sequential model; see
  [docs/operator_fusion.md](../docs/operator_fusion.md)

- Hough compression: `hough_quantize(w, line_bits, t_bits)` returns a
  `HoughCode` (`.stream()`, `.lines`, `.nbytes`, `.decode()`,
  `.decode_on_device()`); see `ptensor.hough`
- Quadratic / cubic quantisation: `poly_quantize(w, degree, bits)` returns a
  `PolyCode` (`.stream()`, `.coeffs`, `.lut()`, `.nbytes`, `.decode()`,
  `.decode_on_device()`); see `ptensor.polyquant`
- Quantisation-aware training (needs TensorFlow): `QATModel(model, mode)` with
  `report_current()`, `report_estimate()`, `fit()`, `report_final()`,
  `export()`; modes in `ptensor.qat.MODES` (float64/32/16, int16, int8,
  uint2..uint7, hough2/3/4, quad2..quad8/quad16, cubic2..cubic8/cubic16). `ptensor.ptm` writes the result as a `.ptm` model.
- Structured pruning (needs TensorFlow): `PrunedModel(model, ratio)` removes
  whole Conv2D filters and Dense units (ratio per model or `{layer: ratio}`)
  with the same `report_current()`, `report_estimate()`, `fit()`,
  `report_final()` steps; `.model` is a plain smaller Keras model that can go
  straight into `QATModel`. `ptensor.prune.sensitivity()` shows which layers
  tolerate pruning. See `ptensor.prune`.

## Generating C++ for a model

`ptensor.codegen.CodeGenerator` turns a trained model (Keras, TFLite, TF
SavedModel; PyTorch and ONNX are registered placeholders) into a
`<name>.h` / `<name>.cpp` / `<name>.cmake` that runs it on pTensor, with
flags for quantisation, fusion, pruning and memory planning:

```python
from ptensor.codegen import CodeGenerator
gen = CodeGenerator("examples/mnist/mnist_int8.tflite", name="mnist")
gen = CodeGenerator(keras_model, weights="hough4", fuse=True, memory="planned")
gen = CodeGenerator(keras_model, activations="int8", calibration_data=x_train[:500])
gen.generate("firmware/model")
print(gen.report())
```

or `python -m ptensor.codegen model.tflite -o out --name mnist`. See
[docs/code_generation.md](../docs/code_generation.md).

## MNIST example

`examples/mnist/mnist_ptensor.py` runs a full-integer int8 TFLite MNIST model
through these bindings and checks it against TFLite. See
[examples/mnist](../examples/mnist/README.md).
[examples/mnist_modes](../examples/mnist_modes/README.md) trains the same
network in every storage mode with QAT and benchmarks each on pTensor.
[examples/mnist_pruning](../examples/mnist_pruning/README.md) does the same for
structured pruning, alone and combined with quantisation.

## Tests

```sh
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

They also run under `ctest` when configured with `-DPTENSOR_BUILD_C_API=ON
-DPACKAGE_TESTS=ON`.
