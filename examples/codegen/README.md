# Code generator examples

`verify_models.py` exercises [`ptensor.codegen`](../../docs/code_generation.md)
end to end: it trains (or loads) an MNIST CNN and a small residual network,
generates pTensor C++ for them in 24 modes (float / packed / Hough /
polynomial weights, full int8, fused or not, pruned, planned or dynamic
memory) from Keras, TFLite and a SavedModel, compiles every one, runs it on
the MNIST test set and checks it against Python. [RESULTS.md](RESULTS.md) has
the numbers.

```sh
# from the repository root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON && cmake --build build -j
pip install tensorflow tflite numpy
PYTHONPATH=python python3 examples/codegen/verify_models.py --out /tmp/cg -n 1000 --cmake-check
```

`--modes a,b` runs a subset (names in `modes()`), `--json` saves the table,
`--cmake-check` also builds a few modes the way firmware would, through the
generated `.cmake` with pTensor's unused features off.

Generating code for your own model is one call:

```python
from ptensor.codegen import CodeGenerator
gen = CodeGenerator("my_model.tflite", name="my_model")   # or a Keras model, SavedModel dir
gen.generate("firmware/model")
print(gen.report())
```

or `PYTHONPATH=python python3 -m ptensor.codegen my_model.tflite -o firmware/model`.
