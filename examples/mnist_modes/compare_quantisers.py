"""Compare Hough line-codebook quantisation with scalar quantisers, before any
training, on the trained MNIST weights and on synthetic controls.

    PYTHONPATH=python python3 examples/mnist_modes/compare_quantisers.py

Uses out/baseline.weights.h5 from train_all_modes.py when it exists (needs
TensorFlow to read it); otherwise only the synthetic controls are run.
Prints a Markdown table of mean squared error at equal storage.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from ptensor import hough, qat  # noqa: E402


def mse(a, b):
    return float(((np.asarray(a, np.float64) - b) ** 2).mean())


def minmax_uniform(w, bits):
    L = 1 << bits
    lo, hi = float(w.min()), float(w.max())
    q = np.rint((w - lo) / (hi - lo) * (L - 1)) * (hi - lo) / (L - 1) + lo
    return mse(q, w)


def lloyd_max(w, bits, iters=60):
    """Optimal non-uniform scalar quantiser (k-means in 1-D). Not a pTensor
    storage mode; included as the best any per-value scalar code can do."""
    x = w.ravel().astype(np.float64)
    L = 1 << bits
    c = np.quantile(x, (np.arange(L) + 0.5) / L)
    for _ in range(iters):
        a = np.abs(x[:, None] - c).argmin(1)
        for j in range(L):
            if (a == j).any():
                c[j] = x[a == j].mean()
    return mse(c[np.abs(x[:, None] - c).argmin(1)], x)


def rows_for(name, w, view=None):
    out = []
    for bits, configs in [(2, [(2, 2), (3, 1)]), (3, [(4, 2), (3, 3)]), (4, [(4, 4), (6, 2)])]:
        uni = qat.UniformFormat(bits).quantizer(w.shape).encode(w)
        cells = {
            "bits": bits,
            "minmax": minmax_uniform(w, bits),
            "uniform": mse(uni.decode(), w),
            "lloyd": lloyd_max(w, bits),
        }
        best, best_fits = None, False
        for lb, tb in configs:
            code = hough.HoughQuantizer(lb, tb, view_shape=view).encode(w)
            e = mse(code.decode(), w)
            cand = (e, lb, tb, code.bits_per_value, code.stride)
            # Only compare at (nearly) equal storage: the codebook counts.
            fits = code.bits_per_value <= bits + 0.35
            if best is None or (fits and (not best_fits or e < best[0])) or \
                    (not fits and not best_fits and code.bits_per_value < best[3]):
                best, best_fits = cand, fits
        cells["hough"] = best
        out.append((name, cells))
    return out


def main():
    rng = np.random.default_rng(0)
    tensors = []
    path = os.path.join(HERE, "out", "baseline.weights.h5")
    if os.path.exists(path):
        sys.path.insert(0, HERE)
        import tensorflow as tf
        from train_all_modes import build_model
        m = build_model(tf)
        m.load_weights(path)
        dense = m.get_layer("dense").kernel.numpy()
        conv2 = np.transpose(m.get_layer("conv2").kernel.numpy(), (3, 0, 1, 2))
        tensors.append(("MNIST dense 784x32", dense, (7, 7, 16, 32)))
        tensors.append(("MNIST conv2 16x3x3x8", conv2, None))
    g = rng.standard_normal(25088).astype(np.float32) * 0.05
    tensors.append(("i.i.d. Gaussian (control)", g.reshape(784, 32), None))
    x = rng.standard_normal(12544)
    y = 0.9 * x + np.sqrt(1 - 0.81) * rng.standard_normal(12544)
    c = (np.stack([x, y], 1).ravel() * 0.05).astype(np.float32)
    tensors.append(("correlated pairs, r=0.9 (control)", c.reshape(784, 32), None))

    print("| Tensor | Bits/weight | Uniform min/max | Uniform fitted (pTensor uintN) | Lloyd-Max scalar (ideal) | Hough (best config) |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for name, w, view in tensors:
        for _, r in rows_for(name, w, view):
            e, lb, tb, bpv, stride = r["hough"]
            print("| %s | %d | %.3g | %.3g | %.3g | %.3g (L=%d, t=%d, %.2f bits incl. codebook, %.0f%% of Lloyd-Max) |" % (
                name, r["bits"], r["minmax"], r["uniform"], r["lloyd"], e, 1 << lb, 1 << tb, bpv,
                100.0 * e / r["lloyd"]))


if __name__ == "__main__":
    main()
