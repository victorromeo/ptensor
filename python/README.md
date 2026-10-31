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

Available: `matmul`, `fully_connected`, `add`, `relu`, `relu6`, `sigmoid`,
`tanh`, `softmax`, `conv2d`, `maxpool2d`, `avgpool2d`, `quantize`,
`dequantize`, `fully_connected_int8`, `pack_bits`, `decode_packed`,
`packed_size`, `float_to_half`, `half_to_float`.

## Tests

```sh
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

They also run under `ctest` when configured with `-DPTENSOR_BUILD_C_API=ON
-DPACKAGE_TESTS=ON`.
