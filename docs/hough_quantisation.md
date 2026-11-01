# Hough line-codebook quantisation

`HoughRomTensor` is a weight storage mode that compresses a tensor with a
codebook of straight lines found by a Hough transform. It is optional at
compile time (`-DPTENSOR_WITH_HOUGH=ON`, the default) and touches no existing
tensor type or operator, so nothing else gets bigger or slower.

## The idea

A Hough transform finds the straight lines that a cloud of points lies on: each
point votes for every line through it in a (θ, ρ) parameter grid, and the
cells with the most votes are the lines. Hough quantisation applies that to
weights:

1. **Pair the weights.** Values *i* and *i + s* form a point (x, y) in the
   plane. The pair stride *s* is chosen per tensor from its shape (1, the last
   dimension, the product of the last two, ...), so that pairs can be
   neighbouring input channels, neighbouring filter taps, or neighbouring
   pixels feeding a dense layer. The encoder tries each and keeps the stride
   whose points fit lines best.
2. **Hough transform.** Points vote in a 48 × (32–256) (θ, ρ) accumulator. The
   strongest peak becomes a line, the points it explains are removed, and the
   vote repeats until there are K = 2^line_bits lines.
3. **Refine (k-lines).** Assign every point to the line, and the position on
   it, that reconstructs it best; refit each line to its points by total least
   squares; fit each line's segment as the best uniform `2^t_bits`-level
   quantiser of the positions along it (outliers are clipped when that lowers
   the error); repeat.
4. **Store** each pair as `line index | position << line_bits` in a bit
   stream, plus the K lines.

A pair costs `line_bits + t_bits` bits, so a weight costs half that, plus a
16 × K byte codebook per tensor.

### Why lines can beat scalar codes

Two weights quantised independently with *b* bits each put their 2^2b
reconstruction points on a rectangular grid aligned with the axes. A line
codebook places the same number of points on K arbitrary lines, each with its
own direction, offset, spacing and extent:

- When paired weights are correlated, the cloud is an ellipse. The Hough
  accumulator peaks at its orientation, and a family of parallel lines across it
  is a rotated grid, which is what transform coding (KLT + scalar quantisation)
  would build. The encoder gets that without being told the data is correlated.
- When they are not correlated, the lines spread out in angle and spacing to
  follow the density of the cloud, which a fixed grid cannot do. This is the
  usual "space-filling and shape" advantage of 2-D vector quantisation over
  scalar quantisation.
- Each line has its own uniform step, so the decoder stays cheap: no entropy
  coder and no per-weight table walk.

### What the device does

Each line is stored in decoder form, four floats derived from its Hough
parameters (θ, ρ) and its segment start *t₀* and step:

```
ax = ρ·cosθ − t₀·sinθ      dx = −step·sinθ
ay = ρ·sinθ + t₀·cosθ      dy =  step·cosθ
```

so decoding a pair is one bit-stream read, one table lookup and two
multiply-adds: `x = ax + tc·dx`, `y = ay + tc·dy`. `HoughRomTensor` decodes on
the fly as operators read it (keeping the last decoded pair, since both halves
are often read close together), and `decode_all()` decodes a whole tensor into
RAM once at start-up for the dense fast kernels.

## Format

| Field | Size |
| --- | --- |
| `line_bits` (0–8), `t_bits` (≥ 1, `line_bits + t_bits ≤ 16`), `pair_stride` (u16) | 4 bytes |
| Lines: `ax, ay, dx, dy` float32 for each of 2^line_bits lines | 16 · 2^line_bits bytes |
| Code stream: one `(line_bits + t_bits)`-bit code per pair, LSB first | ⌈pairs · bits / 8⌉ bytes |

Pairing: values *i* and *i + s* pair up within each block of 2*s* values;
values after the last whole block pair with their neighbour, and an odd final
value pairs with an unused slot. `pTensor/util/hough.hpp` documents and
implements it; `python/ptensor/hough.py` is the encoder and has a NumPy decoder
that matches the C++ one bit for bit (tested through the C API).

```cpp
#include "pTensor/tensors/HoughRomTensor.hpp"
// 16 lines, 16 positions per line: 4 bits per weight + 256 B of lines
Tensor w = new HoughRomTensor({784, 32},
                              HoughEncoding::make(4, 4, /*pair_stride=*/1, lines),
                              code_stream);
```

## Small tensors

The codebook is fixed-size, so on a tensor with a few hundred weights it costs
more than it saves. `qat.HoughFormat` therefore only uses Hough codes when the
codebook is at most 25% of the tensor's storage, and otherwise stores the
tensor as a fitted uniform `uintN` (`PackedRomTensor`) with the same nominal
bits. In the MNIST network only the 784 × 32 dense layer (94% of the weights)
qualifies.

## Results

See [examples/mnist_modes/RESULTS.md](../examples/mnist_modes/RESULTS.md) for
the MNIST table (every pTensor mode, after quantisation-aware training) and the
encoder comparison below for the quantisers on their own.

### The quantiser on its own (no training)

Mean squared error at (nearly) equal storage, codebook included
(`examples/mnist_modes/compare_quantisers.py`). "Uniform fitted" is what
pTensor's `uintN` mode stores (per-tensor affine, offset and step fitted to the
data); Lloyd-Max is the optimal non-uniform scalar quantiser, which pTensor
does not have, shown as the bar any per-value code would have to beat.

| Tensor | Bits/weight | Uniform min/max | Uniform fitted (pTensor uintN) | Lloyd-Max scalar (ideal) | Hough (best config) |
| --- | ---: | ---: | ---: | ---: | --- |
| MNIST dense 784x32 | 2 | 0.00742 | 0.000812 | 0.000741 | 0.000602 (L=8, t=2, 2.04 bits incl. codebook, 81% of Lloyd-Max) |
| MNIST dense 784x32 | 3 | 0.00106 | 0.000321 | 0.000229 | 0.000172 (L=16, t=4, 3.08 bits incl. codebook, 75% of Lloyd-Max) |
| MNIST dense 784x32 | 4 | 0.000229 | 0.000142 | 6.67e-05 | 5.04e-05 (L=64, t=4, 4.33 bits incl. codebook, 76% of Lloyd-Max) |
| MNIST conv2 16x3x3x8 | 2 | 0.00707 | 0.00268 | 0.0022 | 0.00202 (L=4, t=4, 2.47 bits incl. codebook, 92% of Lloyd-Max) |
| MNIST conv2 16x3x3x8 | 3 | 0.00137 | 0.000807 | 0.000669 | 0.000538 (L=8, t=8, 3.92 bits incl. codebook, 80% of Lloyd-Max) |
| MNIST conv2 16x3x3x8 | 4 | 0.000291 | 0.000276 | 0.000212 | 0.000134 (L=16, t=16, 5.81 bits incl. codebook, 63% of Lloyd-Max) |
| i.i.d. Gaussian (control) | 2 | 0.00179 | 0.000374 | 0.000296 | 0.000268 (L=8, t=2, 2.04 bits incl. codebook, 91% of Lloyd-Max) |
| i.i.d. Gaussian (control) | 3 | 0.000305 | 0.000118 | 8.71e-05 | 8.13e-05 (L=16, t=4, 3.08 bits incl. codebook, 93% of Lloyd-Max) |
| i.i.d. Gaussian (control) | 4 | 6.59e-05 | 4.85e-05 | 2.41e-05 | 2.13e-05 (L=64, t=4, 4.33 bits incl. codebook, 88% of Lloyd-Max) |
| correlated pairs, r=0.9 (control) | 2 | 0.00189 | 0.00038 | 0.000296 | 0.000115 (L=8, t=2, 2.04 bits incl. codebook, 39% of Lloyd-Max) |
| correlated pairs, r=0.9 (control) | 3 | 0.00032 | 0.000123 | 8.69e-05 | 3.68e-05 (L=16, t=4, 3.08 bits incl. codebook, 42% of Lloyd-Max) |
| correlated pairs, r=0.9 (control) | 4 | 7e-05 | 5.05e-05 | 2.45e-05 | 1.03e-05 (L=64, t=4, 4.33 bits incl. codebook, 42% of Lloyd-Max) |

- On the MNIST dense layer, where nearly all the weights are, Hough codes have
  19–25% less error than even Lloyd-Max at 2, 3 and 4 bits, and 26–65% less
  than the uniform codes pTensor stores today (the 4-bit Hough row spends
  0.33 extra bits on its 64-line codebook).
- On i.i.d. Gaussian data (no structure for the lines to find) the gain shrinks
  to 7–12%: that part is the generic 2-D vector-quantisation advantage.
- On strongly correlated pairs the gain is 58–61%, because the Hough
  accumulator finds the correlation direction (transform coding for free).
- On conv2 (1,152 weights) the codebook does not pay for itself: the Hough
  rows there use 0.5–1.8 extra bits per weight. That is why the QAT format
  falls back to `uintN` on small tensors.

### After quantisation-aware training (MNIST, accuracy on pTensor)

Three QAT runs per mode from the same float baseline (full 10,000-image test
set, C++ runner):

| Bits per weight | `uintN` size | `uintN` accuracy (3 seeds) | Hough size | Hough accuracy (3 seeds) |
| --- | ---: | --- | ---: | --- |
| 2 | 6,954 B | 96.74 / 97.56 / 97.31 (mean 97.20%) | 7,074 B | 98.12 / 97.99 / 98.03 (mean **98.05%**) |
| 3 | 10,283 B | 98.32 / 98.26 / 98.04 (mean 98.21%) | 10,535 B | 98.28 / 97.90 / 98.14 (mean 98.11%) |
| 4 | 13,612 B | 98.40 / 98.28 / 98.10 (mean 98.26%) | 13,864 B | 98.49 / 98.35 / 98.12 (mean 98.32%) |

The float32 model (106,792 B) scores 98.37%.

### Honest assessment

- **Where it helps:** at 2 bits per weight (15x smaller than float32) Hough
  codes keep the model within 0.3 points of float, beating 2-bit `uintN` by
  0.85 points on average and in every run, for 120 extra bytes (1.7%).
- **Where it doesn't:** at 3 and 4 bits QAT closes the gap. The lower error
  before training does not turn into better accuracy; the differences
  (−0.10 and +0.06 points) are within run-to-run noise, and Hough is 250 bytes
  larger. Use plain `uintN` there.
- **Small tensors:** the 16-byte-per-line codebook only pays off on tensors with
  a few thousand weights or more.
- **Speed:** decoding is a bit-stream read, a lookup and two multiply-adds per
  pair, about the same cost as `PackedRomTensor`'s `uintN` decode (same
  per-image times in RESULTS.md). Like any non-dense tensor it runs operators on
  their reference loops when decoded on the fly; `decode_all()` into RAM once
  gives float32 speed.
- **Encoding cost:** the encoder is NumPy and takes about 0.5 s for the 25k-weight
  dense layer from scratch; QAT re-encodes every step with a warm-started
  codebook, which made each 2-epoch QAT run 10–15 s longer than the `uintN`
  runs here.

So the novel part works and is measurably better than any scalar code before
training, but its practical win after QAT is the 2-bit regime.
