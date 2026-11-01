# MNIST in every pTensor quantisation mode

This example trains one small MNIST CNN in every weight storage mode pTensor
supports, using the Python QAT flow, and then runs each model on pTensor's C++
kernels to measure accuracy, size and inference time.
[RESULTS.md](RESULTS.md) has the latest numbers.

| File | What it is |
| --- | --- |
| `train_all_modes.py` | Trains the float baseline, then for each mode runs the QAT flow (current size → estimated size → quantisation-aware fine-tuning → accuracy and final size), writes a `.ptm` model and benchmarks it with `mnist_modes`. Writes `RESULTS.md` and `results.json`. |
| `mnist_modes.cpp` | C++ runner: loads a `.ptm`, builds `RomTensor` / `PackedRomTensor` / `HoughRomTensor` / `PolyRomTensor` weights straight from it, classifies the test set, prints accuracy, µs per image and weight bytes as JSON. |
| `compare_quantisers.py` | Mean squared error of the Hough codebook against scalar quantisers at equal storage, before any training. |

## Run it

```sh
# from the repository root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON -DPTENSOR_MNIST_RUNNER=ON
cmake --build build -j
pip install tensorflow numpy
PYTHONPATH=python python3 examples/mnist_modes/train_all_modes.py          # ~80 min on 4 cores
PYTHONPATH=python python3 examples/mnist_modes/compare_quantisers.py
```

Options: `--modes hough4,uint4` to run a subset, `--qat-epochs`, `--epochs`
(float pre-training), `--limit N` to time on fewer test images. The float
baseline is cached in `out/baseline.weights.h5`, so later runs fine-tune from
the same starting point.

## Using the QAT flow on your own model

```python
import ptensor
from ptensor import QATModel

q = QATModel(keras_model, "hough4")      # or "cubic4", "quad2", "uint4", "int8", ...
q.report_current()                        # [1/4] size of the float model
q.report_estimate()                       # [2/4] estimated size in the new mode
q.fit(x_train, y_train, epochs=2)         # [3/4] quantisation-aware training
q.report_final(x_test, y_test)            # [4/4] accuracy and actual size
ptensor.ptm.write_float_model("model.ptm", keras_model, q.export())
```

`q.export()` returns each layer's encoded weights (`HoughEncoded`, `PolyEncoded`, `Packed` or
`RawF32`) if you would rather emit C arrays for your firmware than a `.ptm`
file. Modes are listed in `ptensor.qat.MODES`; `HoughFormat(line_bits, t_bits)`
and `UniformFormat(bits)` take any other setting.
