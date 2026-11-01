# PyTorch import: verification results

`verify_pytorch.py` (defaults: 3 training epochs, the first 1,000 MNIST test
images) on an x86-64 container, gcc 14 `-O2`, one inference at a time. Each
model is trained in PyTorch, generated with
`CodeGenerator(model, example_input=...)`, compiled against pTensor and run;
the output is compared with the PyTorch model on the same images.
`pytorch_results.json` has the raw numbers. Accuracy is that of a short
training run; the comparison columns are the point.

| Model | Mode | Nodes | Weights B | Arena B | µs | PyTorch acc. | pTensor acc. | Max abs diff | Same argmax |
|---|---|---|---|---|---|---|---|---|---|
| CNN (conv, batch norm, max pool, dropout, linear) | float, fused, planned | 7 | 106792 | 31360 | 263 | 98.0% | 98.0% | 1.7e-05 | 100% |
| | float, unfused, dynamic | 12 | 107176 | dynamic | 1091 | 98.0% | 98.0% | 1.5e-05 | 100% |
| | int8 PTQ | 7 | 27424 | 7840 | 378 | 98.0% | 98.0% | 1.8 (logits) | 100% |
| MobileNet-style (stride-2 conv, depthwise, hard-swish, squeeze-excitation, ReLU6, global pool) | float, fused, planned | 17 | 5368 | 26944 | 205 | 74.8% | 74.8% | 2.0e-05 | 100% |
| | float, unfused, dynamic | 22 | 6392 | dynamic | 573 | 74.8% | 74.8% | 2.1e-05 | 100% |
| GRU over the 28 image rows, linear head | float, fused, planned | 5 | 46888 | 8512 | 240 | 96.5% | 96.5% | 4.5e-06 | 100% |
| | float, unfused, dynamic | 5 | 46888 | dynamic | 248 | 96.5% | 96.5% | 4.5e-06 | 100% |
| Tiny ViT (4x4 patch conv, positional embedding, pre-norm TransformerEncoderLayer with 3 heads and GELU, layer norm, mean over tokens) | float, fused, planned | 122 | 27032 | 67040 | 1301 | 91.2% | 91.2% | 3.3e-06 | 100% |
| | float, unfused, dynamic | 122 | 27032 | dynamic | 1562 | 91.2% | 91.2% | 3.3e-06 | 100% |

The PyTorch stride-2, padding-1 convolutions in the MobileNet model pad
differently from TF SAME (PyTorch pads 1 on both sides, SAME pads 0 before
and 1 after on a 28-pixel input), so the importer adds a `pad` node in front
of them. The int8 CNN's argmax matches the float PyTorch model on every
image; the logit difference is the int8 quantisation step.

The operator-by-operator checks are in `python/tests/test_pytorch.py`: 128
single-operator and small-model cases, each run through the NumPy
reference interpreter against PyTorch, and compiled and run against the
reference interpreter (fused / planned, and unfused / dynamic for a subset),
plus an int8 model bit-exact with pTensor's kernels through the bindings.
