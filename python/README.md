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

- Hough compression: `hough_quantize(w, line_bits, t_bits)` returns a
  `HoughCode` (`.stream()`, `.lines`, `.nbytes`, `.decode()`,
  `.decode_on_device()`); see `ptensor.hough`
- Quantisation-aware training (needs TensorFlow): `QATModel(model, mode)` with
  `report_current()`, `report_estimate()`, `fit()`, `report_final()`,
  `export()`; modes in `ptensor.qat.MODES` (float64/32/16, int16, int8,
  uint2..uint7, hough2/3/4). `ptensor.ptm` writes the result as a `.ptm` model.

## MNIST example

`examples/mnist/mnist_ptensor.py` runs a full-integer int8 TFLite MNIST model
through these bindings and checks it against TFLite. See
[examples/mnist](../examples/mnist/README.md).
[examples/mnist_modes](../examples/mnist_modes/README.md) trains the same
network in every storage mode with QAT and benchmarks each on pTensor.

## Tests

```sh
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

They also run under `ctest` when configured with `-DPTENSOR_BUILD_C_API=ON
-DPACKAGE_TESTS=ON`.
