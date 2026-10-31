# pTensor

pTensor runs machine-learning inference on very resource-constrained devices
such as microcontrollers. It is a C++11 library with no RTTI, no exceptions,
and no heap allocation in the inference path (memory comes from fixed-size
arena allocators that you size at compile time).

It started as a port of [uTensor](https://github.com/uTensor/uTensor.git) to
resolve local build dependency conflicts and address local project needs.

## Building and testing

```sh
git submodule update --init extern/gtest
cmake -S . -B build -DPACKAGE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `PTENSOR_WITH_QUANTIZED_OPS` | `ON` | Build the int8 TFLM-compatible quantized ops |
| `PTENSOR_WITH_FAST_KERNELS` | `ON` | Pointer-based fast paths for dense tensors (see Performance) |
| `PTENSOR_WITH_SIMPLE_ERROR_HANDLER` | `ON` | Build `SimpleErrorHandler` (uses `std::deque`/`std::function`) |
| `PTENSOR_BARE_METAL` | `ON` for `Generic` targets | Never call `exit()`; halt instead |
| `PTENSOR_SIZE_PROBE` | `OFF` | Build a minimal app used to measure footprint |
| `PTENSOR_BENCH` | `OFF` | Build the host kernel benchmark |
| `PTENSOR_BUILD_C_API` | `OFF` | Build `libptensor_c` (C API, used by the Python package) |
| `PACKAGE_TESTS` | `OFF` | Build the GoogleTest unit tests |
| `PACKAGE_TUTORIALS` | `OFF` | Build the tutorials |
| `PTENSOR_SANITIZE` | `OFF` | Build with AddressSanitizer and UBSan |
| `ENABLE_PRINTING` | `OFF` | Enable debug printing |
| `ENABLE_COVERAGE` | `OFF` | Build with coverage instrumentation |

CI builds and tests every pull request with GCC, Clang, and a sanitizer build,
and cross-compiles for a Cortex-M4 to report code size.

## Python and C

`src/pTensor/c_api/ptensor_c.h` is a plain C interface that runs one operator on
caller-owned buffers. The `python/` package wraps it with NumPy, so you can
verify layers against NumPy or TensorFlow without writing C. See
[python/README.md](python/README.md).

## Layer types

Every operator is a header-only template, so a layer you never use adds
nothing to your firmware (the Cortex-M4 size probe is the same 7.9 KB with or
without these headers included).

| Group | Operators |
| --- | --- |
| Dense / conv | `MatrixMultOperator`, `FullyConnectedOperator`, `Conv2dOperator`, `DepthwiseConv2dOperator` (stride, dilation, depth multiplier), TFLM int8 conv / depthwise / FC |
| Pooling | `MaxPoolOperator`, `MinPoolOperator`, `AvgPoolOperator`, `GlobalAveragePoolOperator` |
| Normalisation | `BatchNormOperator`, `ChannelAffineOperator` (folded batch norm), `SoftmaxOperator` |
| Recurrent | `LstmCellOperator`, `GruCellOperator` (Keras weight layout) |
| Shape | `ReshapeOperator`, `PadOperator`, `ConcatOperator` |
| Elementwise binary (NumPy broadcasting, rank ≤ 4) | `BroadcastAddOperator`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `SquaredDifference`, `Pow` (`XOperator<T>`) |
| Activations | ReLU, ReLU6, Sigmoid, Tanh, LeakyReLU, ELU, SELU, HardSigmoid, HardSwish, Swish/SiLU, GELU, Softplus, Softsign, Clip |
| Math | Abs, Neg, Square, Sqrt, Rsqrt, Exp, Log, Floor, Ceil, Round, Sin, Cos |
| Quantisation | `QuantizeOperator`, `DequantizeOperator` |
| Reductions | `ArgMaxOperator`, `ArgMinOperator` |

Activation and math operators are `UnaryElementwiseOperator<T, Fn>` with a small
functor, so adding your own takes a few lines:

```cpp
struct Mish { float operator()(float x) const { return x * std::tanh(std::log1p(std::exp(x))); } };
using MishOperator = ReferenceOperators::UnaryElementwiseOperator<float, Mish>;
```

## Performance

When every tensor an operator touches is a dense array (`RomTensor`,
`BufferTensor`, `RamTensor`), the operator runs a pointer-based kernel from
`pTensor/ops/fast/` instead of reading each element through a virtual call.
The fast kernels sum in the same order as the reference loops, so results do
not change; `tests/operators/test_extended_ops.cpp` checks this by running
each op twice, once with an input behind a `PackedRomTensor` (which forces the
reference loop). Tensors that are not dense, like packed weights, still work
and use the reference loop.

Host benchmark (`-DPTENSOR_BENCH=ON`, then `build/examples/bench/ptensor_bench`;
x86-64, Release, best of N runs, microseconds):

| Kernel | Reference loop | Fast path | Speedup |
| --- | ---: | ---: | ---: |
| matmul 64x128 @ 128x64 | 7943 | 64 | 123x |
| fully connected 512 -> 256 + ReLU | 1823 | 22 | 82x |
| conv2d 1x32x32x16, 3x3 -> 32, SAME | 72192 | 2122 | 34x |
| depthwise 1x32x32x32, 3x3, SAME | 5594 | 73 | 77x |
| max pool 1x32x32x32, 2x2/2 | 342 | 51 | 7x |
| mul 1x32x32x64 * [64] (broadcast) | 1529 | 13 | 119x |
| add, 65535 elements | 1570 | 8 | 187x |
| swish, 65535 elements | 1207 | 184 | 7x |
| LSTM cell, in 64, hidden 128 | 1463 | 24 | 61x |

Microcontroller gains are smaller than these (no wide SIMD), but the per-element
virtual call and index arithmetic the fast path removes are the bulk of the
reference loop's cost on any target. `-DPTENSOR_WITH_FAST_KERNELS=OFF` drops the
fast paths when flash matters more than speed (about 300 bytes in the size
probe, more in models that use more operators).

## Compressed weight storage

`PackedRomTensor` keeps model constants compressed in flash and decodes each
element as operators read it, so any operator can consume packed weights
unchanged.

| Storage format | Bytes per element |
| --- | --- |
| `PackedEncoding::bits(n, scale, zp)`, n = 1..16 (e.g. 6- or 7-bit uint) | n / 8 |
| `Int8`, `UInt8` | 1 |
| `Int16`, `UInt16`, `Float16` (IEEE half) | 2 |
| `Int32`, `Float32` | 4 |
| `Float64` (double) | 8 |

Integer codes map to real values as `(code - zp) * scale`. The tensor's logical
type (what operators see) is any pTensor type: float tensors get the real value,
and integer tensors get it rounded and saturated.

```cpp
#include "pTensor/tensors/PackedRomTensor.hpp"
// 6-bit weights, real = (code - 32) * 0.02, read by ops as float
Tensor w = new PackedRomTensor({16, 8}, flt,
                               PackedEncoding::bits(6, 0.02f, 32), packed_weights);
```

`pTensor/util/packing.hpp` has the matching host-side helpers
(`pack_bits`, `float_to_half`) for producing packed arrays.

## Keeping the footprint small

- Operators are header-only templates. An operator you never instantiate adds
  nothing to your binary, so include only the operator headers you use rather
  than the `pTensor.h` umbrella header.
- Library code is compiled with `-ffunction-sections -fdata-sections`. Link
  your firmware with `-Wl,--gc-sections` to drop anything you do not call.
- The library never uses the heap. Arenas keep their block table inside the
  allocator object (`localCircularArenaAllocator<Bytes, IndexType, MaxBlocks>`,
  default `MaxBlocks` 32, or set `-DPTENSOR_ARENA_DEFAULT_MAX_BLOCKS=N`).
- Without `ENABLE_PRINTING`, no stdio is linked.
- `-DPTENSOR_WITH_FAST_KERNELS=OFF` keeps only the reference loops.

### Cross-compiling for a microcontroller

```sh
cmake -S . -B build-m4 -DCMAKE_BUILD_TYPE=MinSizeRel -DPTENSOR_SIZE_PROBE=ON \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-cortex-m4.cmake
cmake --build build-m4
arm-none-eabi-size build-m4/examples/size_probe/ptensor_size_probe
```

The size probe (a 2x2 uint8 matmul + add, including the C runtime startup)
is about 7.9 KB of flash on a Cortex-M4.

## Acknowledgements

Acknowledgements to [uTensor](https://github.com/uTensor/uTensor.git).
