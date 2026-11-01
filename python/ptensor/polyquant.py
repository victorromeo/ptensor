"""Polynomial (quadratic / cubic) quantisation of weights (the encoder).

Linear quantisation spaces its 2**bits levels evenly:

    real = (k - zero_point) * scale

Polynomial quantisation bends that line into a polynomial of degree 2
(quadratic) or 3 (cubic) in the normalised code t = 2k / (L - 1) - 1:

    real = c0 + c1 t + c2 t^2 + c3 t^3          (L = 2**bits, t in [-1, 1])

so the levels can be dense where most weights are (near zero) and sparse in
the tails, or lopsided when the weights are skewed. It costs 4 bytes per extra
coefficient per tensor, and the device decodes through a lookup table of the
L level values, so reading a weight is a bit-stream read and a table lookup,
the same work as linear decoding (src/pTensor/util/polyquant.hpp,
PolyRomTensor). With a cubic and 2-bit codes the four levels are free, i.e.
the result is the optimal (Lloyd-Max) 4-level codebook.

Fitting (per tensor, NumPy only so quantisation-aware training can call it
every step):

  1. start from the best uniform (linear) quantiser and from a polynomial
     fitted to the weights' sorted values (density matching),
  2. Lloyd iterations: assign every weight to its nearest level, then refit
     the coefficients by least squares on (t of its code, weight); keep the
     lowest error. Each step can only lower the error, so the result is never
     worse than the linear quantiser it started from.

`PolyCode.decode()` reproduces the device decoder bit for bit (float32,
same operation order), and `decode_on_device()` checks it against the C++
implementation through the C API.
"""
import numpy as np

from .hough import _fit_uniform

HEADER_BYTES = 2  # degree, bits

# Lookup tables bigger than this are not worth storing (16-bit codes would
# need a 256 KB table); such tensors evaluate the polynomial per read.
MAX_LUT_BITS = 8


def code_to_t(bits):
    """t for every code, computed in float32 exactly as the device does."""
    k = np.arange(1 << bits, dtype=np.float32)
    step = np.float32(2.0) / np.float32((1 << bits) - 1)
    return (k * step - np.float32(1.0)).astype(np.float32)


def build_lut(coeffs, degree, bits):
    """The value of every code (float32 Horner, the device's order)."""
    c = np.asarray(coeffs, np.float32)
    t = code_to_t(bits)
    r = np.full_like(t, c[degree])
    for i in range(degree - 1, -1, -1):
        r = (r * t).astype(np.float32) + c[i]
    return r.astype(np.float32)


def _nearest(w, lut):
    """Index of the nearest level for every value (any level order)."""
    order = np.argsort(lut, kind="stable")
    s = lut[order].astype(np.float64)
    mids = 0.5 * (s[1:] + s[:-1])
    return order[np.searchsorted(mids, w)]


class PolyCode(object):
    """One tensor in polynomial form."""

    def __init__(self, shape, degree, bits, coeffs, codes, use_lut):
        self.shape = tuple(int(d) for d in shape)
        self.degree = int(degree)
        self.bits = int(bits)
        self.coeffs = np.zeros(4, np.float32)
        self.coeffs[:self.degree + 1] = np.asarray(coeffs, np.float32)[:self.degree + 1]
        self.codes = np.asarray(codes, np.int64).reshape(-1)
        self.use_lut = bool(use_lut)

    @property
    def count(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def lut(self):
        return build_lut(self.coeffs, self.degree, self.bits)

    @property
    def stream_bytes(self):
        return (self.count * self.bits + 7) // 8

    @property
    def lut_bytes(self):
        return 4 << self.bits if self.use_lut else 0

    @property
    def params_bytes(self):
        """Coefficients and header (always stored)."""
        return 4 * (self.degree + 1) + HEADER_BYTES

    @property
    def nbytes(self):
        """Flash: codes + coefficients + the lookup table if one is stored."""
        return self.stream_bytes + self.params_bytes + self.lut_bytes

    def stream(self):
        from .qat import _pack_bits
        return _pack_bits(self.codes, self.bits)

    def decode(self):
        return self.lut()[self.codes].reshape(self.shape)

    def decode_on_device(self):
        """Decode through the C++ PolyRomTensor (needs the ptensor_c library)."""
        import ctypes
        from ._lib import lib
        from .ops import _check
        stream = np.ascontiguousarray(self.stream())
        out = np.empty(self.count, np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        lut = np.ascontiguousarray(self.lut()) if self.use_lut else None
        _check("decode_poly", lib.ptensor_decode_poly_f32(
            self.degree, self.bits, self.coeffs.ctypes.data_as(fp),
            None if lut is None else lut.ctypes.data_as(ctypes.c_void_p),
            stream.ctypes.data_as(ctypes.c_void_p), self.count,
            out.ctypes.data_as(fp)))
        return out.reshape(self.shape)


def estimate_bytes(count, degree, bits, use_lut=None):
    if use_lut is None:
        use_lut = bits <= MAX_LUT_BITS
    return (count * bits + 7) // 8 + 4 * (degree + 1) + HEADER_BYTES + (4 << bits if use_lut else 0)


def _linear_init(w, bits, degree):
    """The linear quantiser UniformFormat would pick (same scale and zero
    point), written as a degree-1 polynomial in t."""
    levels = 1 << bits
    t0, step = _fit_uniform(w, levels)
    if not step > 0:
        step = max(abs(t0), 1e-8) / max(1, levels - 1)
    scale = float(np.float32(step))
    zp = float(np.rint(-t0 / scale))
    half = 0.5 * (levels - 1)
    c = np.zeros(degree + 1)
    c[0], c[1] = (half - zp) * scale, half * scale
    return c


def _density_init(w, bits, degree):
    """Polynomial through the sorted weights: level density follows the data."""
    s = np.sort(w)
    n = s.size
    t = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    # Keep the extreme tails from bending the fit.
    keep = slice(n // 200, n - n // 200) if n >= 400 else slice(None)
    V = np.vander(t[keep], degree + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, s[keep], rcond=None)
    return c


def _compander_init(w, bits, degree):
    """Polynomial through the high-resolution optimal compander: level density
    proportional to the cube root of the weight density (Panter-Dite)."""
    lo, hi = float(w.min()), float(w.max())
    if not hi > lo:
        return None
    h, edges = np.histogram(w, bins=256, range=(lo, hi))
    g = np.cbrt(h.astype(np.float64))
    G = np.concatenate([[0.0], np.cumsum(g)])
    G = G / G[-1] * 2.0 - 1.0
    t = np.linspace(-1, 1, 512)
    x = np.interp(t, G, edges)
    V = np.vander(t, degree + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, x, rcond=None)
    return c


class PolyQuantizer(object):
    """Encodes one tensor; keeps its coefficients between calls so
    quantisation-aware training can re-encode cheaply (warm start)."""

    def __init__(self, degree, bits, use_lut=None, iters=20):
        if not 1 <= degree <= 3:
            raise ValueError("degree must be 1..3")
        if not 1 <= bits <= 16:
            raise ValueError("bits must be 1..16")
        self.degree = int(degree)
        self.bits = int(bits)
        self.use_lut = bits <= MAX_LUT_BITS if use_lut is None else bool(use_lut)
        self.iters = iters
        self.coeffs = None
        self.t = code_to_t(self.bits).astype(np.float64)

    def _err(self, w, c):
        lut = build_lut(np.asarray(c, np.float32), self.degree, self.bits)
        codes = _nearest(w, lut)
        d = lut[codes].astype(np.float64) - w
        return float(d @ d), codes

    def _lloyd(self, w, c, iters):
        """Lloyd iterations (assign, then least-squares refit), each step
        over-relaxed along the refit direction when that lowers the error
        further (plain Lloyd crawls when there are many levels)."""
        c = np.asarray(c, np.float64)
        best_e, best_codes = self._err(w, c)
        best_c = c
        V_all = np.vander(self.t, self.degree + 1, increasing=True)
        for _ in range(iters):
            new_c, *_ = np.linalg.lstsq(V_all[best_codes], w, rcond=None)
            e, codes = self._err(w, new_c)
            step = new_c - best_c
            cand = (e, new_c, codes)
            alpha = 2.0
            while e < best_e and alpha <= 64:
                c2 = best_c + alpha * step
                e2, codes2 = self._err(w, c2)
                if e2 >= cand[0]:
                    break
                cand = (e2, c2, codes2)
                alpha *= 2
            if cand[0] >= best_e * (1 - 1e-9):
                if cand[0] < best_e:
                    best_e, best_c, best_codes = cand
                break
            best_e, best_c, best_codes = cand
        return best_e, best_c, best_codes

    def encode(self, w, refit_iters=None):
        w32 = np.asarray(w, np.float32)
        flat = w32.astype(np.float64).ravel()
        iters = self.iters if refit_iters is None else refit_iters
        if self.coeffs is None:
            starts = [_linear_init(flat, self.bits, self.degree)]
            if self.degree >= 2 and flat.size > self.degree + 1:
                starts.append(_density_init(flat, self.bits, self.degree))
                comp = _compander_init(flat, self.bits, self.degree)
                if comp is not None:
                    starts.append(comp)
            best = min((self._lloyd(flat, c, max(iters, self.iters)) for c in starts),
                       key=lambda r: r[0])
        else:
            best = self._lloyd(flat, self.coeffs, iters)
            # Re-anchor occasionally lost levels: never be worse than linear.
            if refit_iters is None or refit_iters >= self.iters:
                lin = self._lloyd(flat, _linear_init(flat, self.bits, self.degree), self.iters)
                if lin[0] < best[0]:
                    best = lin
        _, c, codes = best
        self.coeffs = np.asarray(c, np.float64)
        return PolyCode(w32.shape, self.degree, self.bits, np.asarray(c, np.float32),
                        codes, self.use_lut)


def poly_quantize(w, degree, bits, use_lut=None, iters=20):
    """Quantise `w` with a degree-`degree` polynomial of `bits`-bit codes."""
    return PolyQuantizer(degree, bits, use_lut, iters).encode(w)
