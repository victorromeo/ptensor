"""Weight storage formats and quantisation-aware training (QAT).

Every pTensor weight storage mode is a `WeightFormat` here:

    FloatFormat("f32")          RomTensor, float32 (the uncompressed baseline)
    FloatFormat("f64"/"f16")    PackedRomTensor FLOAT64 / FLOAT16
    UniformFormat(bits)         PackedRomTensor UINT_BITS (1..16 bits, affine)
    UniformFormat(8, "int8")    PackedRomTensor INT8 (also "int16")
    HoughFormat(lb, tb)         HoughRomTensor (Hough line codebook)

A format turns a float array into an `Encoded` tensor (what goes in flash) and
can fake-quantise an array (encode, then decode exactly as the device will),
which is what training sees.

QAT flow (`QATModel`, needs TensorFlow/Keras):

    q = QATModel(keras_model, HoughFormat(4, 4))
    q.report_current()           # 1. size of the float model
    q.report_estimate()          # 2. estimated size in the new format
    q.fit(x_train, y_train, epochs=2)   # 3. fine-tune with quantised weights
    q.report_final(x_test, y_test)      # 4. accuracy and actual new size
    encoded = q.export()         # per-layer Encoded tensors for the device

Training uses the straight-through estimator: each step runs forward and
backward with the quantised weights, then applies the gradient to the float
"shadow" weights. Biases stay float32 (they are a few hundred bytes).
"""
import numpy as np

from .hough import HoughQuantizer, estimate_bytes as _hough_bytes, _fit_uniform

# PackedFormat ids (must match pTensor::PackedFormat / ptensor_packed_format).
UINT_BITS, INT8, UINT8, INT16, UINT16, INT32, FLOAT16, FLOAT32, FLOAT64 = range(9)


# ---------------------------------------------------------------------------
# Encoded tensors
# ---------------------------------------------------------------------------

class Encoded(object):
    """Base: a tensor as stored on the device."""
    kind = None

    def __init__(self, shape):
        self.shape = tuple(int(d) for d in shape)

    @property
    def count(self):
        return int(np.prod(self.shape)) if self.shape else 1


class RawF32(Encoded):
    kind = "f32"

    def __init__(self, values):
        values = np.ascontiguousarray(values, np.float32)
        Encoded.__init__(self, values.shape)
        self.values = values

    @property
    def nbytes(self):
        return self.values.nbytes

    def decode(self):
        return self.values

    def describe(self):
        return "float32"


class Packed(Encoded):
    """PackedRomTensor data: format, bit width, affine scale/zero point."""
    kind = "packed"

    def __init__(self, shape, fmt, bits, scale, zero_point, payload, codes=None):
        Encoded.__init__(self, shape)
        self.fmt = int(fmt)
        self.bits = int(bits)
        self.scale = float(np.float32(scale))
        self.zero_point = int(zero_point)
        self.payload = np.ascontiguousarray(payload, np.uint8).reshape(-1)
        self.codes = codes

    @property
    def params_bytes(self):
        return 0 if self.fmt in (FLOAT16, FLOAT32, FLOAT64) else 8

    @property
    def nbytes(self):
        return self.payload.size + self.params_bytes

    def decode(self):
        """Float32 values exactly as a float PackedRomTensor reads them."""
        if self.fmt == FLOAT16:
            return self.payload.view("<f2").astype(np.float32).reshape(self.shape)
        if self.fmt == FLOAT32:
            return self.payload.view("<f4").astype(np.float32).reshape(self.shape)
        if self.fmt == FLOAT64:
            return self.payload.view("<f8").astype(np.float32).reshape(self.shape)
        codes = self.codes.astype(np.float32)
        out = (codes - np.float32(self.zero_point)) * np.float32(self.scale)
        return out.astype(np.float32).reshape(self.shape)

    def describe(self):
        names = {UINT_BITS: "uint%d" % self.bits, INT8: "int8", UINT8: "uint8",
                 INT16: "int16", UINT16: "uint16", FLOAT16: "float16",
                 FLOAT32: "float32", FLOAT64: "float64"}
        return names.get(self.fmt, "packed")


class HoughEncoded(Encoded):
    kind = "hough"

    def __init__(self, code):
        Encoded.__init__(self, code.shape)
        self.code = code

    @property
    def nbytes(self):
        return self.code.nbytes

    def decode(self):
        return self.code.decode()

    def describe(self):
        c = self.code
        return "hough L=%d t=%d stride=%d" % (1 << c.line_bits, 1 << c.t_bits, c.stride)


def _pack_bits(codes, bits):
    codes = np.ascontiguousarray(codes, np.uint32).reshape(-1)
    b = ((codes[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(b.ravel(), bitorder="little")


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

class WeightFormat(object):
    name = "?"

    def quantizer(self, shape, view_shape=None):
        raise NotImplementedError

    def estimate_bytes(self, shape):
        raise NotImplementedError


class _Stateless(object):
    def __init__(self, fmt):
        self.fmt = fmt

    def fake(self, w, refit=True):
        return self.encode(w).decode()

    def encode(self, w):
        return self.fmt._encode(np.asarray(w, np.float32))


class FloatFormat(WeightFormat):
    def __init__(self, kind="f32"):
        if kind not in ("f16", "f32", "f64"):
            raise ValueError(kind)
        self.kind = kind
        self.name = {"f16": "float16", "f32": "float32", "f64": "float64"}[kind]

    def quantizer(self, shape, view_shape=None):
        return _Stateless(self)

    def _encode(self, w):
        if self.kind == "f32":
            return RawF32(w)
        dt, fmt = ("<f2", FLOAT16) if self.kind == "f16" else ("<f8", FLOAT64)
        payload = np.ascontiguousarray(w, dt).view(np.uint8)
        return Packed(w.shape, fmt, 0, 1.0, 0, payload)

    def estimate_bytes(self, shape):
        return int(np.prod(shape)) * {"f16": 2, "f32": 4, "f64": 8}[self.kind]


class UniformFormat(WeightFormat):
    """Per-tensor affine integers: real = (code - zero_point) * scale.

    The scale and offset are fitted to minimise the squared error (outliers are
    clipped when that helps), not just set from min/max.
    """

    def __init__(self, bits, container="bits"):
        self.bits = int(bits)
        self.container = container
        if container == "bits":
            if not 1 <= self.bits <= 16:
                raise ValueError("bits must be 1..16")
            self.name = "uint%d" % self.bits
        elif container in ("int8", "int16"):
            self.bits = 8 if container == "int8" else 16
            self.name = container
        else:
            raise ValueError(container)

    def quantizer(self, shape, view_shape=None):
        return _Stateless(self)

    def _encode(self, w):
        levels = 1 << self.bits
        flat = w.astype(np.float64).ravel()
        t0, step = _fit_uniform(flat, levels)
        if not step > 0:
            step = max(abs(t0), 1e-8) / max(1, levels - 1)
        scale = float(np.float32(step))
        zp = int(np.rint(-t0 / scale))
        codes = np.clip(np.rint(flat / scale) + zp, 0, levels - 1).astype(np.int64)
        if self.container == "bits":
            return Packed(w.shape, UINT_BITS, self.bits, scale, zp,
                          _pack_bits(codes, self.bits), codes)
        half = levels // 2
        signed = codes - half
        dt, fmt = ("<i1", INT8) if self.bits == 8 else ("<i2", INT16)
        payload = np.ascontiguousarray(signed, dt).view(np.uint8)
        return Packed(w.shape, fmt, self.bits, scale, zp - half, payload, signed)

    def estimate_bytes(self, shape):
        return (int(np.prod(shape)) * self.bits + 7) // 8 + 8


class _HoughTensor(object):
    def __init__(self, fmt, shape, view_shape):
        self.fmt = fmt
        self.q = HoughQuantizer(fmt.line_bits, fmt.t_bits, iters=fmt.iters,
                                view_shape=view_shape)

    def fake(self, w, refit=True):
        return self.q.encode(w, refit_iters=1 if refit else 0).decode()

    def encode(self, w):
        return HoughEncoded(self.q.encode(w, refit_iters=self.fmt.iters))


class HoughFormat(WeightFormat):
    """Hough line codebook: (line_bits + t_bits) / 2 bits per weight plus a
    16 * 2^line_bits byte codebook per tensor.

    Tensors too small for the codebook to pay off (it would be more than
    `max_codebook_share` of the tensor's storage) use UniformFormat with the
    same nominal bits instead.
    """

    def __init__(self, line_bits=4, t_bits=4, iters=12, max_codebook_share=0.25):
        self.line_bits = int(line_bits)
        self.t_bits = int(t_bits)
        self.iters = iters
        self.max_codebook_share = max_codebook_share
        self.name = "hough%d+%d" % (self.line_bits, self.t_bits)
        self.fallback = UniformFormat(int(np.ceil((self.line_bits + self.t_bits) / 2.0)))

    @property
    def nominal_bits(self):
        return (self.line_bits + self.t_bits) / 2.0

    def uses_hough(self, shape):
        total = _hough_bytes(int(np.prod(shape)), self.line_bits, self.t_bits)
        book = (1 << self.line_bits) * 16
        return book <= self.max_codebook_share * total

    def quantizer(self, shape, view_shape=None):
        if not self.uses_hough(shape):
            return self.fallback.quantizer(shape)
        return _HoughTensor(self, shape, view_shape)

    def estimate_bytes(self, shape):
        if not self.uses_hough(shape):
            return self.fallback.estimate_bytes(shape)
        return _hough_bytes(int(np.prod(shape)), self.line_bits, self.t_bits)


MODES = {
    "float64": lambda: FloatFormat("f64"),
    "float32": lambda: FloatFormat("f32"),
    "float16": lambda: FloatFormat("f16"),
    "int16": lambda: UniformFormat(16, "int16"),
    "int8": lambda: UniformFormat(8, "int8"),
    "uint7": lambda: UniformFormat(7),
    "uint6": lambda: UniformFormat(6),
    "uint5": lambda: UniformFormat(5),
    "uint4": lambda: UniformFormat(4),
    "uint3": lambda: UniformFormat(3),
    "uint2": lambda: UniformFormat(2),
    "hough4": lambda: HoughFormat(4, 4),
    "hough3": lambda: HoughFormat(4, 2),
    "hough2": lambda: HoughFormat(2, 2),
}


def get_format(name):
    """A WeightFormat by mode name (see MODES), e.g. 'hough4' or 'uint4'."""
    if isinstance(name, WeightFormat):
        return name
    return MODES[name]()


# ---------------------------------------------------------------------------
# Keras QAT
# ---------------------------------------------------------------------------

def _fmt_bytes(n):
    return "%d B (%.1f KB)" % (n, n / 1024.0)


class QATModel(object):
    """Quantisation-aware training of a Keras model's Conv2D/Dense kernels."""

    def __init__(self, model, fmt, view_shapes=None, log=print):
        import tensorflow as tf  # noqa: F401  (fail early with a clear error)
        self.model = model
        self.fmt = get_format(fmt)
        self.log = log
        self.entries = []  # (layer, kernel var, bias var or None, quantizer, device shape)
        view_shapes = view_shapes or {}
        for layer in model.layers:
            cls = type(layer).__name__
            if cls not in ("Conv2D", "Dense"):
                continue
            k = layer.kernel
            dev_shape = _device_shape(cls, tuple(k.shape))
            q = self.fmt.quantizer(dev_shape, view_shapes.get(layer.name))
            self.entries.append((layer, k, getattr(layer, "bias", None), q, dev_shape))

    # -- sizes --------------------------------------------------------------

    def _bias_bytes(self):
        return sum(int(np.prod(b.shape)) * 4 for _, _, b, _, _ in self.entries if b is not None)

    def current_bytes(self):
        return sum(int(np.prod(k.shape)) * 4 for _, k, _, _, _ in self.entries) + self._bias_bytes()

    def estimated_bytes(self):
        return sum(self.fmt.estimate_bytes(s) for _, _, _, _, s in self.entries) + self._bias_bytes()

    def report_current(self):
        n = sum(int(np.prod(k.shape)) for _, k, _, _, _ in self.entries)
        self.log("[1/4] current model: %d weights in %d layers, float32 = %s"
                 % (n, len(self.entries), _fmt_bytes(self.current_bytes())))
        return self.current_bytes()

    def report_estimate(self):
        est = self.estimated_bytes()
        self.log("[2/4] estimated size as %s: %s (%.1fx smaller)"
                 % (self.fmt.name, _fmt_bytes(est), self.current_bytes() / float(est)))
        return est

    # -- weights ------------------------------------------------------------

    def _to_device(self, layer, w):
        return np.ascontiguousarray(_keras_to_device(type(layer).__name__, w))

    def _from_device(self, layer, w):
        return _device_to_keras(type(layer).__name__, w)

    def quantized_weights(self, refit=True, final=False):
        if final:
            return [self._from_device(layer, np.asarray(e.decode(), np.float32))
                    for layer, e, _ in self.export()]
        out = []
        for layer, k, _, q, _ in self.entries:
            wq = q.fake(self._to_device(layer, k.numpy()), refit)
            out.append(self._from_device(layer, np.asarray(wq, np.float32)))
        return out

    def _swap_in(self, values):
        saved = [k.numpy() for _, k, _, _, _ in self.entries]
        for (_, k, _, _, _), v in zip(self.entries, values):
            k.assign(v)
        return saved

    # -- training -----------------------------------------------------------

    def fit(self, x, y, epochs=1, batch_size=128, learning_rate=5e-4, loss=None, seed=0):
        import tensorflow as tf
        loss_fn = loss or tf.keras.losses.SparseCategoricalCrossentropy()
        opt = tf.keras.optimizers.Adam(learning_rate)
        variables = self.model.trainable_variables
        model = self.model

        @tf.function
        def grads_of(xb, yb):
            with tf.GradientTape() as tape:
                l = loss_fn(yb, model(xb, training=True))
            return l, tape.gradient(l, variables)

        rng = np.random.default_rng(seed)
        n = len(x)
        self.log("[3/4] quantisation-aware training: %d epoch(s), %d steps each"
                 % (epochs, (n + batch_size - 1) // batch_size))
        for ep in range(epochs):
            order = rng.permutation(n)
            total, steps = 0.0, 0
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                saved = self._swap_in(self.quantized_weights(refit=True))
                l, g = grads_of(x[idx], y[idx])
                self._swap_in(saved)  # straight-through: update the float weights
                opt.apply_gradients(zip(g, variables))
                total += float(l)
                steps += 1
            self.log("      epoch %d: loss %.4f" % (ep + 1, total / steps))

    def evaluate(self, x, y, final=True):
        """Accuracy with the weights exactly as the device will decode them."""
        saved = self._swap_in(self.quantized_weights(refit=False, final=final))
        try:
            pred = self.model.predict(x, batch_size=512, verbose=0)
        finally:
            self._swap_in(saved)
        return float((pred.argmax(-1) == y).mean())

    def export(self):
        """[(layer, Encoded kernel (device layout), float32 bias or None)].

        The encoding is computed once per set of weights and cached, so the
        accuracy reported and the tensors exported always match."""
        key = tuple(hash(k.numpy().tobytes()) for _, k, _, _, _ in self.entries)
        if getattr(self, "_export_key", None) == key:
            return self._exported
        out = []
        for layer, k, b, q, _ in self.entries:
            enc = q.encode(self._to_device(layer, k.numpy()))
            out.append((layer, enc, None if b is None else b.numpy().astype(np.float32)))
        self._export_key, self._exported = key, out
        return out

    def report_final(self, x, y):
        enc = self.export()
        size = sum(e.nbytes for _, e, _ in enc) + self._bias_bytes()
        acc = self.evaluate(x, y)
        self.log("[4/4] after QAT as %s: accuracy %.2f%%, size %s (%.1fx smaller than float32)"
                 % (self.fmt.name, 100 * acc, _fmt_bytes(size), self.current_bytes() / float(size)))
        for layer, e, _ in enc:
            self.log("      %-12s %-34s %s" % (layer.name, e.describe(), _fmt_bytes(e.nbytes)))
        return acc, size


# Keras stores conv kernels HWIO; pTensor's conv filter is OHWI. Dense kernels
# are [in, out] in both.

def _device_shape(cls, keras_shape):
    if cls == "Conv2D":
        h, w, i, o = keras_shape
        return (o, h, w, i)
    return keras_shape


def _keras_to_device(cls, w):
    return np.transpose(w, (3, 0, 1, 2)) if cls == "Conv2D" else w


def _device_to_keras(cls, w):
    return np.transpose(w, (1, 2, 3, 0)) if cls == "Conv2D" else w
