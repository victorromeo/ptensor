# MNIST with pTensor (Python)

`mnist_ptensor.py` classifies the MNIST test set with a TFLite model, running
every layer through pTensor's C++ kernels via the Python bindings. It reads
each tensor's data type and quantization from the model and runs the layer in
that type, so the int8 model runs in int8 end to end, the way it would on a
microcontroller.

| File | What it is |
| --- | --- |
| `mnist_int8.tflite` | Full-integer model (32 KB): int8 activations, per-channel int8 weights, int32 biases, int8 input and output. The default. |
| `mnist_f32.tflite` | The same network in float32, for comparison. |
| `mnist_ptensor.py` | The sample: a small TFLite reader that maps each operator to a pTensor call. |
| `mnist_data.py` | Downloads and caches the MNIST IDX files (`~/.cache/ptensor/mnist`, or `$PTENSOR_MNIST_DIR`). |
| `train_mnist.py` | How the models were made (needs TensorFlow). Only needed to retrain. |

The network is Conv 3x3x8 (ReLU) → MaxPool 2 → Conv 3x3x16 (ReLU) → MaxPool 2
→ Dense 32 (ReLU) → Dense 10 → Softmax.

## Run it

```sh
# from the repository root
cmake -S . -B build -DPTENSOR_BUILD_C_API=ON && cmake --build build
pip install numpy tflite
PYTHONPATH=python python3 examples/mnist/mnist_ptensor.py            # int8
PYTHONPATH=python python3 examples/mnist/mnist_ptensor.py --model examples/mnist/mnist_f32.tflite
```

Options: `-n 10000` for the whole test set (default 1000), `--batch`,
`--no-reference`.

If TensorFlow or `ai-edge-litert` is installed, the sample also runs TFLite's
reference kernels on the same images and compares. On the full 10,000-image
test set:

| Model | pTensor accuracy | TFLite accuracy | Same prediction | Output difference |
| --- | --- | --- | --- | --- |
| int8 | 97.81% | 97.81% | 100% | identical up to the softmax; softmax within 9/256 (mean 0.0009 LSB) |
| float32 | 97.85% | 97.85% | 100% | 0.0 |

The int8 convolutions, pooling and dense layers are bit-exact with TFLite's
reference kernels. pTensor's int8 softmax dequantizes, runs the float softmax
and requantizes, while TFLite uses a fixed-point exp, so a few probabilities
differ by a few 1/256 steps; the predicted digit never changes. (TFLite's
default XNNPACK delegate rounds requantization slightly differently from its
reference kernels, which is why the comparison uses the reference kernels.)
