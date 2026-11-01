"""Structured (channel / filter) pruning.

Structured pruning removes whole Conv2D filters and whole Dense units, then
cuts the matching input channels out of the next layer. What is left is an
ordinary, smaller dense network: pTensor runs it with the same kernels and
fast paths as any other model, with no sparse format, no masks and no new
code on the device. It composes with every weight storage mode: prune first,
then hand the smaller model to QATModel.

Flow (`PrunedModel`, needs TensorFlow/Keras), the same four steps as QAT:

    p = PrunedModel(keras_model, 0.5)       # remove 50% of every hidden layer
    p.report_current()                      # 1. size and MACs of the model
    p.report_estimate()                     # 2. size and MACs after pruning
    p.fit(x_train, y_train, epochs=2)       # 3. prune, then fine-tune
    p.report_final(x_test, y_test)          # 4. accuracy and actual size
    small = p.model                         # a plain (smaller) Keras model

    q = QATModel(small, "uint4")            # optional: quantise it as well

`ratio` is the fraction of filters/units removed: one float for every
prunable layer, or {layer name: ratio}. The last weighted layer (the
logits) is never pruned. Filters are ranked by the L1 (default) or L2 norm
of their weights; a following BatchNormalization's |gamma| scales the score.
`fit(..., steps=k)` prunes in k equal rounds with fine-tuning after each
(on the MNIST example it did not beat one-shot pruning at the same epoch
budget; see examples/mnist_pruning).

The array functions (`filter_scores`, `keep_indices`, `prune_dense_chain`)
need only NumPy.
"""
import numpy as np

# Layers that keep the channel layout of their input unchanged.
_PASSTHROUGH = ("InputLayer", "MaxPooling2D", "AveragePooling2D", "ReLU",
                "Activation", "Dropout", "Softmax", "LeakyReLU", "ZeroPadding2D",
                "GlobalAveragePooling2D", "GlobalMaxPooling2D")
_WEIGHTED = ("Conv2D", "Dense")


# ---------------------------------------------------------------------------
# Array-level helpers (NumPy only)
# ---------------------------------------------------------------------------

def filter_scores(kernel, criterion="l1", out_axis=-1):
    """Importance of each output channel of a kernel: the L1 or L2 norm of
    the weights feeding it. Keras Conv2D kernels are HWIO and Dense kernels
    [in, out], so the output axis is the last one by default."""
    k = np.moveaxis(np.asarray(kernel, np.float64), out_axis, 0).reshape(np.shape(kernel)[out_axis], -1)
    if criterion == "l1":
        return np.abs(k).sum(1)
    if criterion == "l2":
        return np.sqrt((k * k).sum(1))
    raise ValueError("criterion must be 'l1' or 'l2', not %r" % (criterion,))


def n_keep(n, ratio, multiple_of=1, min_keep=1):
    """How many of n channels survive pruning `ratio` of them, rounded to
    the nearest multiple of `multiple_of` (but never 0 or more than n)."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1), not %r" % (ratio,))
    k = int(round(n * (1.0 - ratio) / multiple_of)) * multiple_of
    return int(min(n, max(k, min_keep, 1)))


def keep_indices(scores, keep):
    """Indices of the `keep` highest scores, in their original order (ties
    go to the lower index, so the result is deterministic)."""
    scores = np.asarray(scores)
    order = np.lexsort((np.arange(len(scores)), -scores))
    return np.sort(order[:keep])


def prune_dense_chain(weights, ratio, criterion="l1", multiple_of=1):
    """Prune a plain MLP given as [(kernel [in, out], bias [out]), ...]:
    every layer but the last loses `ratio` of its units and the next layer
    loses the matching input rows. Returns (new weights, kept index lists)."""
    out, kept, keep_in = [], [], None
    for i, (k, b) in enumerate(weights):
        k = np.asarray(k)
        b = np.asarray(b)
        if keep_in is not None:
            k = k[keep_in]
        if i == len(weights) - 1:
            out.append((k, b))
            break
        idx = keep_indices(filter_scores(k, criterion), n_keep(k.shape[1], ratio, multiple_of))
        out.append((k[:, idx], b[idx]))
        kept.append(idx)
        keep_in = idx
    return out, kept


# ---------------------------------------------------------------------------
# Keras model surgery
# ---------------------------------------------------------------------------

def _cls(layer):
    return type(layer).__name__


def _weighted(model):
    return [l for l in model.layers if _cls(l) in _WEIGHTED]


def prunable_layers(model):
    """Names of the layers whose outputs can be pruned: every Conv2D and
    Dense except the last weighted layer (the model's outputs)."""
    return [l.name for l in _weighted(model)[:-1]]


def _ratio_for(ratio, name):
    if isinstance(ratio, dict):
        return float(ratio.get(name, 0.0))
    return float(ratio)


def _next_bn(model, layer):
    """The BatchNormalization right after `layer` (through pass-through
    layers), whose |gamma| scales that layer's filter scores, or None."""
    layers = model.layers
    for nxt in layers[layers.index(layer) + 1:]:
        c = _cls(nxt)
        if c == "BatchNormalization":
            return nxt
        if c not in _PASSTHROUGH:
            return None
    return None


def _check_sequential(model):
    if _cls(model) != "Sequential":
        raise NotImplementedError("structured pruning supports Keras Sequential models")


def choose_channels(model, ratio, criterion="l1", multiple_of=1, keep_counts=None):
    """{layer name: kept output channel indices} for every prunable layer.

    keep_counts, if given, is {layer name: number of channels to keep} and
    overrides `ratio` for those layers (used by the stepped schedule)."""
    _check_sequential(model)
    chosen = {}
    for layer in model.layers:
        if layer.name not in prunable_layers(model):
            continue
        k = layer.kernel.numpy()
        scores = filter_scores(k, criterion)
        bn = _next_bn(model, layer)
        if bn is not None and bn.scale:
            scores = scores * np.abs(bn.gamma.numpy())
        n = k.shape[-1]
        if keep_counts and layer.name in keep_counts:
            keep = int(keep_counts[layer.name])
        else:
            keep = n_keep(n, _ratio_for(ratio, layer.name), multiple_of)
        chosen[layer.name] = keep_indices(scores, keep)
    return chosen


def _clone_layer(layer, **overrides):
    cfg = layer.get_config()
    cfg.update(overrides)
    return type(layer).from_config(cfg)


def apply_pruning(model, chosen):
    """A new, smaller Sequential model with only the channels in `chosen`
    ({layer name: kept output indices}; layers not listed keep all of
    theirs) and the matching weights copied over. `model` is not changed."""
    import tensorflow as tf
    _check_sequential(model)
    new_layers, weights = [], []
    keep = None             # kept channels of the current activation (None = all)
    spatial = None          # (h, w, c) of the map a Flatten flattened, while keep applies
    for layer in model.layers:
        c = _cls(layer)
        if c in ("Conv2D", "Dense"):
            k, b = layer.kernel.numpy(), (layer.bias.numpy() if layer.use_bias else None)
            if keep is not None:
                if c == "Conv2D":
                    k = k[:, :, keep, :]
                elif spatial is not None:
                    h, w, ch = spatial
                    k = k.reshape(h, w, ch, -1)[:, :, keep, :].reshape(h * w * len(keep), -1)
                else:
                    k = k[keep]
            out_keep = chosen.get(layer.name)
            if out_keep is not None:
                k = k[..., out_keep]
                b = None if b is None else b[out_keep]
            n_out = k.shape[-1]
            nl = _clone_layer(layer, **({"filters": n_out} if c == "Conv2D" else {"units": n_out}))
            new_layers.append(nl)
            weights.append((nl, [k] + ([] if b is None else [b])))
            keep, spatial = out_keep, None
        elif c == "BatchNormalization":
            nl = _clone_layer(layer)
            w = layer.get_weights()
            if keep is not None:
                w = [v[keep] for v in w]
            new_layers.append(nl)
            weights.append((nl, w))
        elif c == "DepthwiseConv2D":
            if layer.depth_multiplier != 1 and keep is not None:
                raise NotImplementedError("pruning before a DepthwiseConv2D with depth_multiplier != 1")
            nl = _clone_layer(layer)
            w = layer.get_weights()
            if keep is not None:
                w = [w[0][:, :, keep, :]] + [v[keep] for v in w[1:]]
            new_layers.append(nl)
            weights.append((nl, w))
        elif c == "Flatten":
            nl = _clone_layer(layer)
            new_layers.append(nl)
            if keep is not None:
                spatial = tuple(int(d) for d in layer.input.shape[1:])
                if len(spatial) != 3:
                    raise NotImplementedError("Flatten of a rank-%d map" % len(spatial))
        elif c in _PASSTHROUGH:
            if c == "InputLayer":
                continue
            new_layers.append(_clone_layer(layer))
        else:
            raise NotImplementedError("structured pruning through a %s layer" % c)
    small = tf.keras.Sequential([tf.keras.layers.Input(model.input_shape[1:])] + new_layers,
                                name=model.name + "_pruned")
    for nl, w in weights:
        nl.set_weights(w)
    return small


def prune(model, ratio, criterion="l1", multiple_of=1):
    """One-shot structured pruning: a smaller copy of `model` (no fine-tuning)."""
    return apply_pruning(model, choose_channels(model, ratio, criterion, multiple_of))


# ---------------------------------------------------------------------------
# Size / cost accounting
# ---------------------------------------------------------------------------

def model_stats(model):
    """(weights, bytes as float32 incl. biases, multiply-accumulates per
    inference) over the Conv2D/Dense/DepthwiseConv2D layers."""
    n_w = n_b = macs = 0
    for layer in model.layers:
        c = _cls(layer)
        if c not in ("Conv2D", "Dense", "DepthwiseConv2D"):
            continue
        ws = layer.get_weights()
        k = ws[0]
        n_w += k.size
        n_b += sum(v.size for v in ws[1:])
        if c == "Dense":
            macs += k.size
        else:
            oh, ow = (int(d) for d in layer.output.shape[1:3])
            macs += k.size * oh * ow
    return n_w, 4 * (n_w + n_b), macs


def _fmt(n):
    return "%d B (%.1f KB)" % (n, n / 1024.0)


def _widths(model):
    return ", ".join("%s %d" % (l.name, l.get_weights()[0].shape[-1]) for l in _weighted(model))


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------

class PrunedModel(object):
    """Structured pruning plus fine-tuning of a Keras Sequential model.

    After fit(), `model` is the smaller Keras model (the original passed in
    is left untouched); `kept` maps each pruned layer to the indices of the
    original channels it kept."""

    def __init__(self, model, ratio, criterion="l1", multiple_of=1, log=print):
        import tensorflow as tf  # noqa: F401  (fail early with a clear error)
        _check_sequential(model)
        self.original = model
        self.model = model
        self.ratio = ratio
        self.criterion = criterion
        self.multiple_of = multiple_of
        self.log = log
        self.kept = None
        self._target = {}
        for layer in model.layers:
            if layer.name in prunable_layers(model):
                n = layer.kernel.shape[-1]
                self._target[layer.name] = n_keep(n, _ratio_for(ratio, layer.name), multiple_of)
        if isinstance(ratio, dict):
            unknown = set(ratio) - set(self._target)
            if unknown:
                raise ValueError("not prunable: %s (prunable: %s)"
                                 % (", ".join(sorted(unknown)), ", ".join(prunable_layers(model))))

    # -- reports ------------------------------------------------------------

    def report_current(self):
        n, size, macs = model_stats(self.original)
        self.log("[1/4] current model: %d weights, float32 = %s, %d MACs per inference"
                 % (n, _fmt(size), macs))
        self.log("      widths: %s" % _widths(self.original))
        return size

    def estimated_model(self):
        """The one-shot pruned model (before fine-tuning)."""
        return prune(self.original, self.ratio, self.criterion, self.multiple_of)

    def report_estimate(self):
        small = self.estimated_model()
        n0, s0, m0 = model_stats(self.original)
        n, size, macs = model_stats(small)
        self.log("[2/4] after pruning: %d weights, float32 = %s (%.1fx smaller), %d MACs (%.1fx fewer)"
                 % (n, _fmt(size), s0 / float(size), macs, m0 / float(macs)))
        self.log("      widths: %s" % _widths(small))
        return size

    # -- pruning and fine-tuning --------------------------------------------

    def _schedule(self, steps):
        """Kept channel counts after each of `steps` equal pruning rounds."""
        full = {l.name: l.kernel.shape[-1] for l in self.original.layers if l.name in self._target}
        out = []
        for s in range(1, steps + 1):
            out.append({name: int(round(full[name] - (full[name] - k) * s / float(steps)))
                        for name, k in self._target.items()})
        return out

    def fit(self, x, y, epochs=2, steps=1, batch_size=128, learning_rate=5e-4, loss=None,
            validation_data=None):
        """Prune in `steps` rounds; after each round fine-tune for `epochs`
        epochs. Returns the pruned Keras model (also `self.model`)."""
        import tensorflow as tf
        if steps < 1:
            raise ValueError("steps must be >= 1")
        loss = loss or tf.keras.losses.SparseCategoricalCrossentropy()
        self.log("[3/4] structured pruning in %d step(s), %d fine-tuning epoch(s) after each"
                 % (steps, epochs))
        current = self.original
        index_maps = {}  # layer name -> kept indices into the original channels
        for s, counts in enumerate(self._schedule(steps)):
            chosen = choose_channels(current, 0.0, self.criterion, keep_counts=counts)
            # compose with earlier rounds so `kept` refers to the original model
            for name, idx in chosen.items():
                prev = index_maps.get(name)
                index_maps[name] = idx if prev is None else prev[idx]
            current = apply_pruning(current, chosen)
            current.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss,
                            metrics=["accuracy"])
            self.log("      step %d: widths %s" % (s + 1, _widths(current)))
            if epochs:
                h = current.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0,
                                validation_data=validation_data)
                for e, l in enumerate(h.history["loss"]):
                    self.log("      epoch %d: loss %.4f" % (e + 1, l))
        self.model, self.kept = current, index_maps
        return current

    def evaluate(self, x, y):
        pred = self.model.predict(x, batch_size=512, verbose=0)
        return float((pred.argmax(-1) == y).mean())

    def report_final(self, x, y):
        acc = self.evaluate(x, y)
        _, s0, m0 = model_stats(self.original)
        _, size, macs = model_stats(self.model)
        self.log("[4/4] after pruning + fine-tuning: accuracy %.2f%%, float32 = %s (%.1fx smaller), "
                 "%.1fx fewer MACs" % (100 * acc, _fmt(size), s0 / float(size), m0 / float(macs)))
        return acc, size


def sensitivity(model, x, y, ratios=(0.25, 0.5, 0.75), criterion="l1", log=print):
    """Accuracy after one-shot pruning each prunable layer on its own at
    each ratio (no fine-tuning): {layer: {ratio: accuracy}}. A layer whose
    accuracy holds up can take a larger ratio in a per-layer dict."""
    out = {}
    for name in prunable_layers(model):
        out[name] = {}
        for r in ratios:
            small = prune(model, {name: r}, criterion)
            acc = float((small.predict(x, batch_size=512, verbose=0).argmax(-1) == y).mean())
            out[name][r] = acc
        log("      %-10s %s" % (name, "  ".join("%d%%: %.2f%%" % (100 * r, 100 * a)
                                                   for r, a in out[name].items())))
    return out
