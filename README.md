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
| `PTENSOR_WITH_SIMPLE_ERROR_HANDLER` | `ON` | Build `SimpleErrorHandler` (uses `std::deque`/`std::function`) |
| `PTENSOR_BARE_METAL` | `ON` for `Generic` targets | Never call `exit()`; halt instead |
| `PTENSOR_SIZE_PROBE` | `OFF` | Build a minimal app used to measure footprint |
| `PACKAGE_TESTS` | `OFF` | Build the GoogleTest unit tests |
| `PACKAGE_TUTORIALS` | `OFF` | Build the tutorials |
| `PTENSOR_SANITIZE` | `OFF` | Build with AddressSanitizer and UBSan |
| `ENABLE_PRINTING` | `OFF` | Enable debug printing |
| `ENABLE_COVERAGE` | `OFF` | Build with coverage instrumentation |

CI builds and tests every pull request with GCC, Clang, and a sanitizer build,
and cross-compiles for a Cortex-M4 to report code size.

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
