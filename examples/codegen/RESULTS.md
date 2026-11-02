# Code generator verification results

`verify_models.py --cmake-check -n 1000` on an x86-64 container (gcc 14,
`-O2`), first 1,000 MNIST test images, one inference at a time. The Keras
models were trained by the script itself (2 epochs, so accuracy is not the
point; the comparisons are). `results.json` has the raw numbers.

Columns: **Weights B** is the constant data the generated file holds (flash).
**Arena B** is the planned activation arena (for dynamic mode, the RAM
allocator's high-water mark). **µs** is per inference with the default
full-feature pTensor build. **Check against Python**: float graphs against
the NumPy reference interpreter on the generated graph (weights decoded as
the device decodes them), max absolute difference over all outputs; int8
graphs against pTensor's kernels through the Python bindings, max difference
in int8 steps (the bindings have no int8 depthwise, so the residual int8
model is checked against TFLite only). **Check against original**: the
model the generator started from (Keras / SavedModel output, or TFLite's
reference kernels for `.tflite` inputs), max absolute difference and how
often the predicted digit agrees.

| Mode | What | Source | Accuracy | Weights B | Arena B | µs | Check against Python | Check against original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f32-unfused-dynamic | float32, every node a reference op, RamTensors | keras | 96.8% | 106792 | dynamic (90944 hw) | 514 | NumPy ref 2.0e-06 | 1.9e-06 (argmax 100.0%) |
| f32-fused-planned | float32, fused ops, planned arena (defaults) | keras | 96.8% | 106792 | 31360 | 268 | NumPy ref 2.0e-06 | 1.9e-06 (argmax 100.0%) |
| f32-fused-nofast | as above, `fast_kernels=False` (only changes the .cmake build below) | keras | 96.8% | 106792 | 31360 | 289 | NumPy ref 2.0e-06 | 1.9e-06 (argmax 100.0%) |
| float16 | float16 weights (PackedRomTensor) | keras | 96.8% | 53576 | 31360 | 2928 | NumPy ref 2.4e-06 | 9.1e-04 (argmax 100.0%) |
| int8w | int8 weights, float activations | keras | 96.9% | 26944 | 31360 | 2699 | NumPy ref 2.5e-06 | 3.7e-02 (argmax 99.9%) |
| uint4 | 4-bit weights, decoded from flash | keras | 96.9% | 13628 | 31360 | 2986 | NumPy ref 1.5e-06 | 3.9e-01 (argmax 98.6%) |
| uint4-ram | 4-bit weights, decoded to RAM at init | keras | 96.9% | 13628 | 31360 | 255 | NumPy ref 1.5e-06 | 3.9e-01 (argmax 98.6%) |
| hough4 | Hough codebook weights, from flash | keras | 96.6% | 13872 | 31360 | 2758 | NumPy ref 2.5e-06 | 3.9e-01 (argmax 98.7%) |
| hough4-ram | Hough weights, decoded to RAM at init | keras | 96.6% | 13872 | 31360 | 256 | NumPy ref 2.5e-06 | 3.9e-01 (argmax 98.7%) |
| cubic4 | cubic 4-bit weights (PolyRomTensor) | keras | 96.5% | 13836 | 31360 | 2280 | NumPy ref 2.0e-06 | 4.6e-01 (argmax 98.3%) |
| quad3-unfused | quadratic 3-bit weights, unfused | keras | 96.4% | 10379 | 31360 | 5454 | NumPy ref 3.2e-06 | 6.4e-01 (argmax 98.0%) |
| prune50 | prune 50% of channels (one-shot) | keras | 79.8% | 27176 | 15680 | 115 | NumPy ref 4.2e-07 | 9.1e-01 (argmax 80.9%) |
| prune50-uint6 | prune 50% + 6-bit weights | keras | 79.0% | 5372 | 15680 | 648 | NumPy ref 3.9e-07 | 9.1e-01 (argmax 79.9%) |
| int8-ptq | full int8, calibrated from 500 images | keras | 96.9% | 27424 | 7840 | 356 | bindings 0 LSB | 4.0e-02 (argmax 99.9%) |
| int8-ptq-floatio-dyn | full int8 with float in/out, RamTensors | keras | 96.9% | 27424 | dynamic (15776 hw) | 379 | – | 4.0e-02 (argmax 99.9%) |
| prune25-int8-ptq | prune 25% + full int8 | keras | 93.3% | 15678 | 5880 | 267 | bindings 0 LSB | 7.6e-01 (argmax 95.2%) |
| tflite-int8-repo | examples/mnist/mnist_int8.tflite | tflite | 97.7% | 27424 | 7840 | 354 | bindings 0 LSB | TFLite ref 5 (argmax 100.0%) |
| tflite-f32-repo | examples/mnist/mnist_f32.tflite, unfused | tflite | 97.7% | 106792 | 31360 | 403 | NumPy ref 1.9e-06 | TFLite ref 0 (argmax 100.0%) |
| tflite-int8 | TFLite converter int8 of the Keras model | tflite | 96.9% | 27424 | 7840 | 339 | bindings 0 LSB | TFLite ref 4 (argmax 100.0%) |
| savedmodel | SavedModel export of the Keras model | tf | 96.8% | 106792 | 31360 | 261 | NumPy ref 2.0e-06 | 1.9e-06 (argmax 100.0%) |
| resnet-fused | residual net, fused (BN folded) | keras | 93.2% | 17576 | 12544 | 60 | NumPy ref 9.8e-07 | 1.4e-06 (argmax 100.0%) |
| resnet-unfused-dyn | residual net, unfused, RamTensors | keras | 93.2% | 17832 | dynamic (56448 hw) | 314 | NumPy ref 9.5e-07 | 1.4e-06 (argmax 100.0%) |
| resnet-hough4 | residual net, Hough weights | keras | 93.4% | 2676 | 12544 | 379 | NumPy ref 9.2e-07 | 3.8e-01 (argmax 97.1%) |
| resnet-tflite-int8 | residual net via TFLite int8 (int8 depthwise + add) | tflite | 93.3% | 4944 | 3136 | 346 | – | TFLite ref 0 (argmax 100.0%) |
| act-int8-ptq | avg pool + tanh + sigmoid net, generator int8 PTQ | keras | 93.1% | 7216 | 7840 | 640 | bindings 0 LSB | 3.5e-02 (argmax 99.3%) |
| act-tflite-int8 | same net via TFLite int8 (int8 average pool, tanh, logistic) | tflite | 92.9% | 7216 | 7840 | 589 | bindings 0 LSB | TFLite ref 0 (argmax 100.0%) |

Observations

* With float32 weights the generated code matches Keras, TFLite and the
  SavedModel to within 2e-6. Encoded weights (float16 / int8 / uint4 /
  Hough / cubic / quad) match the NumPy reference of the encoded graph to
  within 3.3e-6, so the generator and the device decode them identically;
  the difference from Keras is the quantisation itself.
* Every int8 mode is bit-exact with the Python bindings. Against TFLite's
  reference kernels the int8 models agree exactly up to the softmax; the
  softmax is within 4-5/256 on a few images (TFLite uses a fixed-point
  exp) and never changes the predicted digit.
* `fuse=True` halves inference time for the MNIST CNN (268 vs 514 µs) and
  cuts the residual net from 314 to 60 µs (batch norm folded away).
* Decoding compressed weights from flash costs ~10x (2.3-3 ms); decoding
  them to RAM at init (`weights_in_ram=True`) gets float speed back while
  keeping flash at 13.6 KB instead of 107 KB.
* Planned arena: 31,360 B float / 7,840 B int8 for the CNN, the same as
  pTensor's ArenaPlanner (examples/memory_plan). Dynamic RamTensors peak at
  90,944 B for the float CNN.
* One-shot pruning without fine-tuning loses ~17 points here; use
  `ptensor.prune.PrunedModel` to prune with training and pass the result to
  the generator (examples/mnist_pruning gets 97.75% at 50%).

## Built through the generated .cmake

Each of these modes was also built as a firmware project would build it:
`include(<name>.cmake)` (forces the needed pTensor features ON, the rest
OFF), `add_subdirectory(ptensor)`, link. The output is compared with the
full-feature build above.

| Mode | pTensor features ON | Same output as full build | µs | Program size B (incl. weights) |
| --- | --- | --- | --- | --- |
| f32-unfused-dynamic | FAST_KERNELS | yes | 520 | 170640 |
| f32-fused-planned | FAST_KERNELS, FUSION | yes | 268 | 165040 |
| f32-fused-nofast | FUSION | yes | 4317 | 156480 |
| hough4-ram | FAST_KERNELS, FUSION, HOUGH | yes | 270 | 79416 |
| cubic4 | FAST_KERNELS, FUSION, POLY_QUANT | yes | 2051 | 74473 |
| tflite-int8-repo | FAST_KERNELS, QUANTIZED_OPS | yes | 376 | 90833 |
| resnet-tflite-int8 | FAST_KERNELS, FUSION, QUANTIZED_OPS | yes | 341 | 85783 |

`f32-fused-nofast` shows the cost of leaving out the fast kernels: 4.3 ms
instead of 0.27 ms, for 8.5 KB less code.
