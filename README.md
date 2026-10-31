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
| `PACKAGE_TESTS` | `OFF` | Build the GoogleTest unit tests |
| `PACKAGE_TUTORIALS` | `OFF` | Build the tutorials |
| `PTENSOR_SANITIZE` | `OFF` | Build with AddressSanitizer and UBSan |
| `ENABLE_PRINTING` | `OFF` | Enable debug printing |
| `ENABLE_COVERAGE` | `OFF` | Build with coverage instrumentation |

CI builds and tests every pull request with GCC, Clang, and a sanitizer build.

## Acknowledgements

Acknowledgements to [uTensor](https://github.com/uTensor/uTensor.git).
