"""Hough line-codebook compression for weights (the encoder).

Idea
----
Group a weight tensor's values into pairs and treat each pair as a point
(x, y) in the plane. A Hough transform over those points finds the straight
lines that most points lie close to. Keep 2**line_bits of them as a codebook,
and store every pair as

    (which line, where along it)  ->  line_bits + t_bits bits per pair

so the cost is (line_bits + t_bits) / 2 bits per weight plus 16 bytes per
line. Decoding on the device is a table lookup and two multiply-adds per pair
(see src/pTensor/util/hough.hpp and HoughRomTensor).

Why lines: when paired values are correlated (neighbouring filter taps,
neighbouring pixels feeding a dense layer) the points form an elongated cloud.
The Hough accumulator finds its orientation, and a family of parallel lines
across it is a rotated quantisation grid (transform coding) whose spacing the
encoder chooses per line. When the values are not correlated the lines spread
out in angle instead. The encoder picks the pairing (pair_stride) whose points
fit lines best.

Encoding steps (all here, on the host):
  1. pair the values with each candidate stride,
  2. Hough transform: vote in a (theta, rho) accumulator and take peaks one
     at a time, removing the points each peak explains,
  3. refine with k-lines clustering: assign each pair to the line and step
     that reconstruct it best, refit each line by total least squares, repeat,
  4. quantise the position along each line uniformly over its segment.

Everything here is NumPy only, so quantisation-aware training can call it on
every step. `HoughCode.decode()` reproduces the device decoder bit for bit
(float32 arithmetic), and `decode_on_device()` checks it against the C++
implementation through the C API.
"""
import numpy as np

LINE_FLOATS = 4  # ax, ay, dx, dy
HEADER_BYTES = 4  # line_bits, t_bits, pair_stride (u16)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def pair_count(count, stride):
    block = 2 * stride
    full = count // block
    tail = count - full * block
    return full * stride + (tail + 1) // 2


def pair_indices(count, stride):
    """(i0, i1) flat indices of each pair; i1 is -1 for the unused slot."""
    block = 2 * stride
    full = count // block
    b = np.arange(full)[:, None] * block
    r = np.arange(stride)[None, :]
    i0 = (b + r).ravel()
    i1 = i0 + stride
    t0 = np.arange(full * block, count, 2)
    t1 = t0 + 1
    t1[t1 >= count] = -1
    return np.concatenate([i0, t0]).astype(np.int64), np.concatenate([i1, t1]).astype(np.int64)


def candidate_strides(shape, count):
    """Pair strides worth trying: products of trailing dimensions."""
    out = [1]
    s = 1
    for d in reversed(tuple(shape)[1:]):
        s *= int(d)
        if 2 * s <= count and s <= 0xFFFF and s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

class _Lines(object):
    """Lines in Hough form: unit normal n=(cos t, sin t), offset rho, and the
    quantised segment [t0, t0 + (L-1)*step] along direction d=(-sin, cos)."""

    def __init__(self, theta, rho, t0, step):
        self.theta = np.asarray(theta, np.float64)
        self.rho = np.asarray(rho, np.float64)
        self.t0 = np.asarray(t0, np.float64)
        self.step = np.asarray(step, np.float64)

    def copy(self):
        return _Lines(self.theta.copy(), self.rho.copy(), self.t0.copy(), self.step.copy())

    @property
    def k(self):
        return self.theta.size

    def decoder_floats(self):
        """(K, 4) float32: ax, ay, dx, dy as stored for the device."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        ax = self.rho * c - self.t0 * s
        ay = self.rho * s + self.t0 * c
        dx = -self.step * s
        dy = self.step * c
        return np.stack([ax, ay, dx, dy], 1).astype(np.float32)


def _hough_detect(pts, n_lines, n_theta=48, n_rho=None):
    """Classic Hough voting with peak removal. Returns (theta, rho) arrays."""
    radius = float(np.sqrt((pts ** 2).sum(1)).max()) * 1.0001 + 1e-12
    if n_rho is None:
        n_rho = int(np.clip(4 * np.sqrt(n_lines) * 4, 32, 256))
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    bin_w = 2.0 * radius / n_rho
    remaining = np.ones(len(pts), bool)
    out_t, out_r = [], []
    cols = np.arange(n_theta)
    for _ in range(n_lines):
        p = pts[remaining]
        if len(p) == 0:
            break
        rho = p[:, :1] * cos_t + p[:, 1:] * sin_t
        rb = np.clip(((rho + radius) / bin_w).astype(np.int64), 0, n_rho - 1)
        acc = np.bincount((rb * n_theta + cols).ravel(), minlength=n_rho * n_theta)
        peak = int(acc.argmax())
        r_i, t_i = divmod(peak, n_theta)
        th = thetas[t_i]
        rh = (r_i + 0.5) * bin_w - radius
        out_t.append(th)
        out_r.append(rh)
        d = np.abs(pts[:, 0] * np.cos(th) + pts[:, 1] * np.sin(th) - rh)
        remaining &= d > bin_w
    # Not enough distinct peaks (tiny tensors): repeat with small offsets.
    while len(out_t) < n_lines:
        j = len(out_t) % max(1, len(out_t))
        out_t.append(out_t[j] if out_t else 0.0)
        out_r.append((out_r[j] if out_r else 0.0) + bin_w * (len(out_t) / n_lines))
    return np.array(out_t), np.array(out_r)


def _segments(pts, theta, rho, assign, k, levels, prev=None):
    """Fit each line to its points (total least squares) and set its segment."""
    theta = theta.copy()
    rho = rho.copy()
    t0 = np.zeros(k) if prev is None else prev.t0.copy()
    step = np.zeros(k) if prev is None else prev.step.copy()
    for j in range(k):
        m = assign == j
        cnt = int(m.sum())
        if cnt == 0:
            continue
        p = pts[m]
        if cnt >= 2:
            c = p.mean(0)
            cov = np.cov((p - c).T) if cnt > 2 else np.outer(p[1] - p[0], p[1] - p[0])
            w, v = np.linalg.eigh(cov)
            d = v[:, -1]  # main direction
            if w[-1] > 1e-20:
                th = np.arctan2(d[0], -d[1])  # normal = d rotated by -90 deg
                theta[j] = th
                rho[j] = c[0] * np.cos(th) + c[1] * np.sin(th)
        else:
            rho[j] = p[0, 0] * np.cos(theta[j]) + p[0, 1] * np.sin(theta[j])
        t = -p[:, 0] * np.sin(theta[j]) + p[:, 1] * np.cos(theta[j])
        t0[j], step[j] = _fit_uniform(t, levels)
    return _Lines(theta, rho, t0, step)


def _fit_uniform(t, levels, iters=8):
    """Offset and step of the `levels`-point uniform quantiser that best fits
    the positions `t` (Lloyd iterations restricted to uniform spacing, so
    outliers are clipped when that lowers the error)."""
    lo, hi = float(t.min()), float(t.max())
    if levels == 1 or hi - lo <= 0:
        return (0.5 * (lo + hi) if levels == 1 else lo), 0.0
    t0, step = lo, (hi - lo) / (levels - 1)
    best = None
    for _ in range(iters):
        c = np.clip(np.rint((t - t0) / step), 0, levels - 1)
        err = float(((t0 + c * step - t) ** 2).sum())
        if best is not None and err >= best[0] - 1e-18:
            break
        best = (err, t0, step)
        # Least squares for (t0, step) given the codes.
        n = c.size
        sc, scc = c.sum(), (c * c).sum()
        st, sct = t.sum(), (c * t).sum()
        det = n * scc - sc * sc
        if det <= 0:
            break
        new_step = (n * sct - sc * st) / det
        if not new_step > 0:
            break
        t0 = (st - new_step * sc) / n
        step = new_step
    return best[1], best[2]


def _assign(pts, lines, levels, chunk=1 << 15):
    """Best (line, t code) per point and its squared error."""
    fl = lines.decoder_floats().astype(np.float64)
    ax, ay, dx, dy = fl[:, 0], fl[:, 1], fl[:, 2], fl[:, 3]
    dd = dx * dx + dy * dy
    safe = np.where(dd > 0, dd, 1.0)
    best_k = np.empty(len(pts), np.int64)
    best_c = np.empty(len(pts), np.int64)
    best_e = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        rx = p[:, :1] - ax
        ry = p[:, 1:] - ay
        tc = np.where(dd > 0, (rx * dx + ry * dy) / safe, 0.0)
        tc = np.clip(np.rint(tc), 0, levels - 1)
        ex = rx - tc * dx
        ey = ry - tc * dy
        e = ex * ex + ey * ey
        k = e.argmin(1)
        idx = np.arange(len(p))
        best_k[s:s + chunk] = k
        best_c[s:s + chunk] = tc[idx, k]
        best_e[s:s + chunk] = e[idx, k]
    return best_k, best_c, best_e


class HoughCode(object):
    """A tensor encoded as a Hough line codebook plus (line, t) codes."""

    def __init__(self, shape, line_bits, t_bits, stride, lines, codes):
        self.shape = tuple(int(d) for d in shape)
        self.line_bits = int(line_bits)
        self.t_bits = int(t_bits)
        self.stride = int(stride)
        self.lines = np.ascontiguousarray(lines, np.float32)  # (K, 4)
        self.codes = np.ascontiguousarray(codes, np.uint32)  # (pairs,)

    @property
    def count(self):
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def code_bits(self):
        return self.line_bits + self.t_bits

    def stream(self):
        """The packed code stream (LSB-first, code_bits per pair)."""
        bits = ((self.codes[:, None] >> np.arange(self.code_bits, dtype=np.uint32)) & 1).astype(np.uint8)
        return np.packbits(bits.ravel(), bitorder="little")

    @property
    def stream_bytes(self):
        return (len(self.codes) * self.code_bits + 7) // 8

    @property
    def codebook_bytes(self):
        return self.lines.size * 4

    @property
    def nbytes(self):
        """Total storage: code stream + codebook + 4-byte header."""
        return self.stream_bytes + self.codebook_bytes + HEADER_BYTES

    @property
    def bits_per_value(self):
        return 8.0 * self.nbytes / self.count

    def decode(self):
        """Float32 values exactly as HoughRomTensor decodes them."""
        mask = np.uint32((1 << self.line_bits) - 1)
        k = (self.codes & mask).astype(np.int64)
        tc = (self.codes >> np.uint32(self.line_bits)).astype(np.float32)
        L = self.lines
        x = L[k, 0] + tc * L[k, 2]
        y = L[k, 1] + tc * L[k, 3]
        out = np.zeros(self.count, np.float32)
        i0, i1 = pair_indices(self.count, self.stride)
        out[i0] = x
        ok = i1 >= 0
        out[i1[ok]] = y[ok]
        return out.reshape(self.shape)

    def decode_on_device(self):
        """Decode through the C++ HoughRomTensor (needs the ptensor_c library)."""
        import ctypes
        from ._lib import lib
        from .ops import _check
        stream = np.ascontiguousarray(self.stream())
        out = np.empty(self.count, np.float32)
        _check("decode_hough", lib.ptensor_decode_hough_f32(
            self.line_bits, self.t_bits, self.stride,
            self.lines.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            stream.ctypes.data_as(ctypes.c_void_p), self.count,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return out.reshape(self.shape)


def estimate_bytes(count, line_bits, t_bits):
    """Storage for `count` values (exact for any stride but 1 slot of padding)."""
    pairs = (count + 1) // 2
    return (pairs * (line_bits + t_bits) + 7) // 8 + (1 << line_bits) * LINE_FLOATS * 4 + HEADER_BYTES


class HoughQuantizer(object):
    """Encodes one tensor; keeps its codebook between calls so quantisation-
    aware training can re-encode cheaply (warm start) on every step."""

    def __init__(self, line_bits=4, t_bits=4, iters=12, view_shape=None, strides=None):
        if not (0 <= line_bits <= 8 and 1 <= t_bits and line_bits + t_bits <= 16):
            raise ValueError("need 0 <= line_bits <= 8, t_bits >= 1, line_bits + t_bits <= 16")
        self.line_bits = line_bits
        self.t_bits = t_bits
        self.iters = iters
        self.view_shape = view_shape
        self.strides = strides
        self.stride = None
        self.lines = None

    @property
    def k(self):
        return 1 << self.line_bits

    @property
    def levels(self):
        return 1 << self.t_bits

    def _points(self, flat, stride):
        i0, i1 = pair_indices(flat.size, stride)
        x = flat[i0]
        y = np.where(i1 >= 0, flat[np.maximum(i1, 0)], 0.0)
        return np.stack([x, y], 1)

    def _fit(self, pts, iters, lines=None):
        k, levels = self.k, self.levels
        if lines is None:
            theta, rho = _hough_detect(pts, k)
            # Initial assignment: nearest line by perpendicular distance.
            d = np.abs(pts[:, :1] * np.cos(theta) + pts[:, 1:] * np.sin(theta) - rho)
            assign = d.argmin(1)
            lines = _segments(pts, theta, rho, assign, k, levels)
        best = None
        for _ in range(max(1, iters)):
            kk, cc, ee = _assign(pts, lines, levels)
            err = float(ee.sum())
            if best is None or err < best[0]:
                best = (err, lines.copy(), kk, cc)
            # Re-seed unused lines at the worst-reconstructed points.
            used = np.bincount(kk, minlength=k)
            for j in np.where(used == 0)[0]:
                w = int(ee.argmax())
                kk[w] = j
                ee[w] = 0.0
            lines = _segments(pts, lines.theta, lines.rho, kk, k, levels, prev=lines)
        kk, cc, ee = _assign(pts, lines, levels)
        if float(ee.sum()) < best[0]:
            best = (float(ee.sum()), lines.copy(), kk, cc)
        return best

    def encode(self, w, refit_iters=None):
        """Encode array `w`; returns a HoughCode."""
        w = np.asarray(w, np.float64)
        flat = w.ravel()
        shape = w.shape
        if self.lines is None or self.stride is None:
            view = self.view_shape if self.view_shape is not None else shape
            strides = self.strides or candidate_strides(view, flat.size)
            results = []
            for s in strides:
                pts = self._points(flat, s)
                results.append((self._fit(pts, max(3, self.iters // 3)), s))
            (_, _, _, _), self.stride = min(results, key=lambda r: r[0][0])
            pts = self._points(flat, self.stride)
            err, lines, kk, cc = self._fit(pts, self.iters)
        else:
            pts = self._points(flat, self.stride)
            iters = self.iters if refit_iters is None else refit_iters
            err, lines, kk, cc = self._fit(pts, iters, lines=self.lines)
        self.lines = lines
        codes = (kk.astype(np.uint32) | (cc.astype(np.uint32) << np.uint32(self.line_bits)))
        return HoughCode(shape, self.line_bits, self.t_bits, self.stride,
                         lines.decoder_floats(), codes)

    def __call__(self, w, refit_iters=1):
        """Fake-quantise: encode and decode (for quantisation-aware training)."""
        return self.encode(w, refit_iters=refit_iters).decode()

    def nbytes(self, count):
        return estimate_bytes(count, self.line_bits, self.t_bits)


def hough_quantize(w, line_bits=4, t_bits=4, iters=12, view_shape=None):
    """One-shot: encode `w` and return its HoughCode."""
    return HoughQuantizer(line_bits, t_bits, iters, view_shape).encode(w)
