# Structured pruning on MNIST

`prune_mnist.py` prunes the [mnist_modes](../mnist_modes/README.md) CNN with
`ptensor.prune.PrunedModel`, fine-tunes it, optionally quantises the smaller
model with `QATModel`, and runs every result on pTensor with the mnist_modes
C++ runner. [RESULTS.md](RESULTS.md) has the numbers.

Structured pruning removes whole filters and units, so the pruned model is an
ordinary smaller network: the device runs it with the existing dense kernels
and fast paths, and there is no new code, format or build option on the C++
side (nothing to compile in or out, and nothing slower for unpruned models).

## Run it

```sh
# from the repository root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON -DPTENSOR_MNIST_RUNNER=ON
cmake --build build -j
pip install tensorflow numpy
PYTHONPATH=python python3 examples/mnist_pruning/prune_mnist.py     # ~15 min on 4 cores
```

It reuses the float baseline cached by mnist_modes (`examples/mnist_modes/out/`),
training it first if it is missing. `--runs prune50,prune50+uint4` runs a subset.

## What the numbers say

- **Pruning buys speed; quantisation buys size.** Removing half the channels
  makes inference 2.4x faster (130 vs 311 µs) for 0.6 points (97.75% vs
  98.37%). Quantised weights decoded on the fly are slower than float32, and
  decoded to RAM they are the same speed, so quantisation alone never speeds
  the model up.
- **At equal size, quantisation is more accurate.** uint4 is 13.6 KB at 98.40%
  where prune50 is 27 KB at 97.75%; hough2 is 7 KB at 98.12% where prune75 is
  7 KB at 94.41%.
- **Together they give the fastest small models.** prune50 + full-integer
  int8 is 7.1 KB, 97.88%, 200 µs: 3.8x smaller and 2.1x faster than unpruned
  int8-full (27 KB, 98.37%, 416 µs). prune50 + uint4 is 3.6 KB at 97.69%,
  prune75 + uint4 is under 1 KB at 94.42%.
- **This network is already small.** Pruning it is no better than training
  the narrower network from scratch (narrow50 97.70% vs prune50 97.75%;
  narrow75 94.96% vs prune75 94.41%), and conv2 and dense lose accuracy fast
  (see the sensitivity table). Pruning pays off most on over-wide models,
  where it also saves the cost of searching for the right widths.
- **Two pruning rounds did not help** at this budget (one epoch after each
  round instead of two after one): 93.20% vs 94.41% at 75%.

## Using it on your own model

```python
from ptensor import PrunedModel, QATModel, ptm, prune

prune.sensitivity(model, x_val, y_val)     # which layers tolerate pruning
p = PrunedModel(model, 0.5)                # or {"conv2": 0.25, "dense": 0.5}
p.report_current()                         # [1/4] size and MACs now
p.report_estimate()                        # [2/4] size and MACs after pruning
p.fit(x_train, y_train, epochs=2)          # [3/4] prune, fine-tune (steps=k for k rounds)
p.report_final(x_test, y_test)             # [4/4] accuracy and size
q = QATModel(p.model, "uint4")             # optional: quantise the smaller model
q.fit(x_train, y_train, epochs=2)
ptm.write_float_model("model.ptm", p.model, q.export())
```

Supported layers: Conv2D and Dense are pruned (all but the last weighted
layer); BatchNormalization, DepthwiseConv2D (depth multiplier 1), Flatten,
pooling, activation, dropout and padding layers follow the pruned channels.
Filters are ranked by L1 norm (`criterion="l2"` for L2), scaled by the
following BatchNormalization's |gamma| when there is one; `multiple_of=4`
rounds kept widths for SIMD-friendly shapes. Keras Sequential models only.
