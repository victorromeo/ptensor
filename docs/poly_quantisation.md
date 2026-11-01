# Quadratic and cubic quantisation

`PolyRomTensor` stores a weight tensor as `bits`-bit integer codes whose
values follow a quadratic or cubic polynomial instead of a straight line, and
decodes them through a lookup table. It is optional at compile time
(`-DPTENSOR_WITH_POLY_QUANT=ON`, the default), and it is a separate tensor
class, so no existing tensor type or operator changes size or speed.

## The idea

Linear (affine) quantisation spaces its `L = 2^bits` levels evenly:

```
real = (k − zero_point) · scale
```

Trained weights are not spread evenly. Most sit near zero and a few sit in long
tails, so a linear grid either clips the tails or wastes most of its levels on
values that hardly occur. Polynomial quantisation bends the line. With the code
normalised to `t = 2k / (L − 1) − 1`, so that `t` runs from −1 to 1:

```
quadratic:  real = c0 + c1·t + c2·t²
cubic:      real = c0 + c1·t + c2·t² + c3·t³
```

The levels are then dense where the slope of the polynomial is small and
sparse where it is large. A cubic with a small `c1` and a large `c3` packs
levels near zero and spreads them out in the tails, which is the shape the
weight distribution asks for. Linear quantisation is the special case
`c2 = c3 = 0`, so the method extends it and can never do worse given the same
fitting.

Quadratic and cubic behave differently on real weights:

- **Cubic** has an odd term, so it can be steep at both ends and flat in the
  middle at the same time. That matches the bell-shaped, roughly symmetric
  weights of a trained layer.
- **Quadratic** is a parabola: it can only make the levels denser on one side
  than the other. It helps when a tensor is skewed and does little when the
  tensor is symmetric, which most are.

With 2-bit codes a cubic has four free coefficients for four levels, so it can
place the levels anywhere: it becomes the optimal (Lloyd-Max) 4-level scalar
quantiser.

## Lookup tables

The polynomial is never evaluated per weight on the device. It is materialised
once per tensor as a table of `L` floats, the value of every code, and decoding
a weight is a bit-stream read and a table index. That is the same work as linear
`PackedRomTensor` decoding (which multiplies instead of indexing), so the
speed is unchanged.

The table costs `4 · 2^bits` bytes per tensor:

| Code width | Table per tensor |
| ---: | ---: |
| 2 bits | 16 B |
| 3 bits | 32 B |
| 4 bits | 64 B |
| 5 bits | 128 B |
| 6 bits | 256 B |
| 7 bits | 512 B |
| 8 bits | 1 KB |
| 16 bits | 256 KB (not stored) |

There are two ways to hold it:

- **In flash**, next to the codes (what the `.ptm` files and the benchmark do).
  Nothing is computed at start-up, and the table counts towards the model size.
- **In RAM**, built at start-up from the 3–4 stored coefficients with
  `polyquant::build_lut()` (or `ptensor_poly_build_lut()` in the C API). Flash
  then holds only the codes and 12–16 bytes of coefficients per tensor, at the
  cost of the table in RAM.

For 16-bit codes a 256 KB table would be larger than the whole model, so the
encoder stores no table and `PolyRomTensor` evaluates the polynomial (three
multiply-adds) for each read. Pass `lut = nullptr` to get the same behaviour at
any width.

The device and the Python encoder build the table with the same float32
operations in the same order, so the two tables are bit-identical; the Python
tests check this through the C API.

## Fitting the polynomial

`python/ptensor/polyquant.py` fits each tensor with NumPy, which is fast
enough to run on every quantisation-aware training step:

1. **Starting points.** The best linear quantiser (the same scale and zero point
   that `UniformFormat` picks); a polynomial through the sorted weights (level
   density follows the weight density); and a polynomial through the
   high-resolution optimal compander, whose level density follows the cube root
   of the weight density (Panter–Dite).
2. **Lloyd iterations.** Assign every weight to its nearest level, then refit
   the coefficients by least squares on `(t of its code, weight)`. Each step can
   only lower the squared error. Plain Lloyd crawls when there are many levels,
   so every refit is also tried at 2×, 4×, 8×, … its step, and the best is
   kept.
3. Keep the best result over all starting points. Because one of them is the
   linear quantiser, the result is never worse than linear.

During quantisation-aware training the weights are trained with the
straight-through estimator, as for the other modes in `ptensor.qat`. The
coefficients are kept between steps and refined with one Lloyd iteration per
step (a warm start), except in the last epoch, where the levels are frozen and
only the codes change; the export keeps the levels the weights were trained
against. Refitting right up to the end, and refitting again at export, made
2–3 bit models noisy (a 2-bit quadratic dropped to 93.6% on one seed); freezing
the last epoch fixed that.

On synthetic weights of the MNIST dense layer's size (25,088 values), the
squared error relative to the best linear quantiser at the same width is:

| Bits | Gaussian, quadratic | Gaussian, cubic | Laplacian, quadratic | Laplacian, cubic |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.80 | 0.79 | 0.94 | 0.85 |
| 3 | 0.80 | 0.74 | 0.90 | 0.69 |
| 4 | 0.60 | 0.50 | 0.68 | 0.43 |
| 6 | 0.74 | 0.51 | 0.76 | 0.31 |
| 8 | 0.95 | 0.58 | 0.94 | 0.38 |
| 16 | 0.99 | 0.61 | 0.99 | 0.51 |

## Format

```
PolyEncoding { u8 degree (1..3), u8 bits (1..16), f32 coeffs[4], const float* lut }
codes:  LSB-first bit stream, `bits` bits per weight (as PackedFormat::UIntBits)
t(k)    = float(k) · (2 / float(2^bits − 1)) − 1            (float32)
real(k) = lut ? lut[k] : ((c3·t + c2)·t + c1)·t + c0         (float32, Horner)
```

In a `.ptm` file (storage id 5): `u8 degree, u8 bits, f32 coeffs[4],
u8 has_lut, f32 lut[2^bits] if has_lut`, then the code stream.

## Using it

```cpp
#include "pTensor.h"
// From the encoder: codes, coefficients and (optionally) the table.
extern const uint8_t dense_codes[];      // 4-bit codes, in flash
extern const float dense_coeffs[4];
extern const float dense_lut[16];        // or build it in RAM:
// static float dense_lut[16]; polyquant::build_lut(dense_coeffs, 3, 4, dense_lut);

Tensor w = new PolyRomTensor({784, 32},
                             PolyEncoding::make(3, 4, dense_coeffs, dense_lut),
                             dense_codes);
```

```python
from ptensor import QATModel
q = QATModel(keras_model, "cubic4")      # quad2..quad8, quad16, cubic2..cubic8, cubic16
q.fit(x_train, y_train, epochs=2)
q.report_final(x_test, y_test)
```

## Results on MNIST

The same CNN and training budget as the other modes (float baseline, then 2
epochs of QAT), measured by the C++ runner on pTensor over all 10,000 test
images. Size is all weight storage in flash, including the lookup tables
(in brackets). Full table: [examples/mnist_modes/RESULTS.md](../examples/mnist_modes/RESULTS.md).

| Code width | Linear | Size | Accuracy | Quadratic size (LUT) | Accuracy | Cubic size (LUT) | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 bits | int16 | 53560 B | 98.44% | 53584 B | 98.39% | 53600 B | 98.41% |
| 8 bits | int8 | 26928 B | 98.40% | 31048 B (4096 B) | 98.42% | 31064 B (4096 B) | 98.42% |
| 7 bits | uint7 | 23599 B | 98.38% | 25671 B (2048 B) | 98.40% | 25687 B (2048 B) | 98.40% |
| 6 bits | uint6 | 20270 B | 98.38% | 21318 B (1024 B) | 98.37% | 21334 B (1024 B) | 98.40% |
| 5 bits | uint5 | 16941 B | 98.31% | 17477 B (512 B) | 98.49% | 17493 B (512 B) | 98.44% |
| 4 bits | uint4 | 13612 B | 98.40% | 13892 B (256 B) | 98.40% | 13908 B (256 B) | 98.22% |
| 3 bits | uint3 | 10283 B | 98.32% | 10435 B (128 B) | 98.30% | 10451 B (128 B) | 98.38% |
| 2 bits | uint2 | 6954 B | 96.74% | 7042 B (64 B) | 97.92% | 7058 B (64 B) | 97.72% |

Mean over three QAT seeds (float32 is 98.37%):

| Bits | Linear (uint) | Quadratic | Cubic | Hough |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 98.26% | 98.33% | 98.22% | 98.32% |
| 3 | 98.21% | 98.14% | 98.10% | 98.11% |
| 2 | 97.20% | 97.92% | 97.91% | 98.05% |

What this shows:

- **2 bits: a real gain.** Quadratic and cubic average 97.9% against 97.2% for
  linear, for 88–104 extra bytes (1.3%). Hough is still slightly better
  (98.05%) at a similar size.
- **3 bits and up: a tie.** Every mode is within the seed-to-seed spread
  (about ±0.2 points), so after QAT the network adapts to a linear grid just as
  well as to a curved one.
- **Before any retraining the curve matters much more.** Post-training
  quantisation to 2 bits gives 77.6% linear, 94.2% quadratic and 96.0% cubic;
  at 3 bits 97.4%, 97.3% and 97.6%. If a model cannot be retrained, cubic is
  the better choice at low widths.
- **Lookup tables are expensive at 6–8 bits.** At 8 bits the four tables add
  4 KB, 15% on top of int8, for no accuracy gain. Build them in RAM, or stay
  linear, at those widths. At 2–5 bits they cost 64–512 bytes.
- **Quadratic matches cubic after QAT** even though cubic has 20–60% lower
  squared error than quadratic on synthetic bell-shaped weights; the network's retraining
  absorbs the difference.
- **Speed is unchanged:** on-the-fly decoding is a bit read and a table index,
  the same cost as linear `PackedRomTensor` (5.3–6.0 ms per image on the
  reference loops on this host, vs 5.9–6.4 ms for uint), and decoding to RAM
  once gives float32 speed.
