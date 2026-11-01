"""Keras (Keras 3 / tf.keras) Sequential and functional models."""
import numpy as np

from ..ir import Graph
from .base import Frontend, FrontendUnavailable, register_frontend

_IDENTITY = ("InputLayer", "Dropout", "SpatialDropout2D", "GaussianNoise",
             "GaussianDropout", "ActivityRegularization", "Identity")
_CLAMPS = {"linear": None, "relu": "relu", "relu6": "relu6"}
_SEPARATE = {"sigmoid": "sigmoid", "tanh": "tanh"}


def _shape(t):
    return tuple(1 if d is None else int(d) for d in t.shape)


def _act_name(layer):
    act = getattr(layer, "activation", None)
    if act is None:
        return "linear"
    return getattr(act, "__name__", str(act))


def _pair(v):
    return (int(v), int(v)) if np.isscalar(v) else (int(v[0]), int(v[1]))


@register_frontend
class KerasFrontend(Frontend):
    name = "keras"
    extensions = (".keras", ".h5", ".hdf5")

    def accepts(self, source):
        if isinstance(source, str):
            return Frontend.accepts(self, source)
        mod = type(source).__module__
        return (mod.startswith("keras") or mod.startswith("tf_keras")) and hasattr(source, "layers")

    def load(self, source, **options):
        if isinstance(source, str):
            try:
                import keras
            except ImportError:
                raise FrontendUnavailable("loading %s needs keras / tensorflow" % source)
            source = keras.models.load_model(source, compile=False)
        return _Importer(source).run()


class _Importer(object):
    def __init__(self, model):
        self.model = model
        self.g = Graph(getattr(model, "name", "model") or "model")
        self.g.source = "keras"
        self.known = {}  # id(KerasTensor) -> IR tensor name

    # -- helpers --------------------------------------------------------------

    def act(self, kt, layer):
        return self.g.add_tensor(layer.name, _shape(kt)).name

    def const(self, layer, what, value):
        return self.g.add_const("%s_%s" % (layer.name, what), np.asarray(value, np.float32)).name

    def apply_activation(self, layer, name, kt, node_attrs):
        """Fold a clamp activation into the node attrs, or return the kind of
        activation that needs a node of its own."""
        a = _act_name(layer)
        if a in _CLAMPS:
            node_attrs["activation"] = _CLAMPS[a]
            return None
        if a in _SEPARATE or a == "softmax":
            node_attrs["activation"] = None
            return a
        raise NotImplementedError("activation %r in layer %s" % (a, layer.name))

    def finish_activation(self, layer, kind, src, kt):
        if kind is None:
            return src
        out = self.g.add_tensor("%s_%s" % (layer.name, kind), _shape(kt)).name
        if kind == "softmax":
            self.g.add_node("softmax", [src], [out], {"beta": 1.0}, name=layer.name + "_softmax")
        else:
            self.g.add_node("activation", [src], [out], {"kind": kind}, name=layer.name + "_" + kind)
        return out

    def bias(self, layer, n):
        if getattr(layer, "use_bias", False) and getattr(layer, "bias", None) is not None:
            return self.const(layer, "bias", layer.bias.numpy())
        return self.const(layer, "bias", np.zeros(n, np.float32))

    # -- the walk ------------------------------------------------------------

    def run(self):
        m = self.model
        if not getattr(m, "inputs", None):
            raise ValueError("build the Keras model (give it an Input) before generating code")
        for kt in m.inputs:
            name = self.g.add_tensor(kt.name if isinstance(kt.name, str) else "input",
                                     _shape(kt)).name
            self.known[id(kt)] = name
            self.g.inputs.append(name)
        for layer, node in self._order():
            cls = type(layer).__name__
            ins = node.input_tensors
            outs = node.output_tensors
            if len(outs) != 1:
                raise NotImplementedError("layer %s has %d outputs" % (layer.name, len(outs)))
            srcs = [self.known[id(t)] for t in ins]
            self.known[id(outs[0])] = self.layer(layer, cls, srcs, ins, outs[0])
        for kt in m.outputs:
            self.g.outputs.append(self.known[id(kt)])
        self.g.remove_unused()
        return self.g

    def _order(self):
        """(layer, node) pairs in execution order, found by walking back from
        the model outputs (Keras keeps one node per call of a layer, and
        Sequential models call their layers more than once)."""
        order, done, seen = [], set(), set(self.known)

        def visit(kt):
            if id(kt) in seen:
                return
            hist = getattr(kt, "_keras_history", None)
            if hist is None:
                raise NotImplementedError("cannot trace Keras tensor %s" % kt)
            layer, ni = hist[0], hist[1]
            node = layer._inbound_nodes[ni]
            if type(layer).__name__ == "InputLayer" or not node.input_tensors:
                raise NotImplementedError("model input %s is not in model.inputs" % layer.name)
            if type(layer).__name__ in ("Functional", "Sequential", "Model"):
                raise NotImplementedError("nested model %s: flatten it first" % layer.name)
            for t in node.input_tensors:
                visit(t)
            if (id(layer), ni) not in done:
                done.add((id(layer), ni))
                order.append((layer, node))
            for t in node.output_tensors:
                seen.add(id(t))

        for kt in self.model.outputs:
            visit(kt)
        return order

    def layer(self, layer, cls, srcs, ins, out):
        g = self.g
        cfg = layer.get_config()
        if cls in _IDENTITY:
            return srcs[0]
        if cfg.get("data_format", "channels_last") not in (None, "channels_last"):
            raise NotImplementedError("%s: only channels_last is supported" % layer.name)

        if cls == "Conv2D":
            if cfg.get("groups", 1) != 1:
                raise NotImplementedError("grouped Conv2D %s" % layer.name)
            k = layer.kernel.numpy()                       # HWIO
            f = self.const(layer, "filter", np.transpose(k, (3, 0, 1, 2)))
            attrs = {"stride": _pair(cfg["strides"]), "padding": cfg["padding"].upper(),
                     "dilation": _pair(cfg.get("dilation_rate", 1))}
            kind = self.apply_activation(layer, None, out, attrs)
            y = g.add_tensor(layer.name, _shape(out)).name
            g.add_node("conv2d", [srcs[0], f, self.bias(layer, k.shape[3])], [y], attrs, layer.name)
            return self.finish_activation(layer, kind, y, out)

        if cls in ("DepthwiseConv2D", "SeparableConv2D"):
            dk = (layer.depthwise_kernel if cls == "SeparableConv2D" else layer.kernel).numpy()
            kh, kw, c, mult = dk.shape
            f = self.const(layer, "dw_filter", dk.reshape(1, kh, kw, c * mult))
            attrs = {"stride": _pair(cfg["strides"]), "padding": cfg["padding"].upper(),
                     "dilation": _pair(cfg.get("dilation_rate", 1)), "depth_multiplier": mult}
            if cls == "SeparableConv2D":
                mid_shape = _shape(out)[:3] + (c * mult,)
                mid = g.add_tensor(layer.name + "_dw", mid_shape).name
                attrs["activation"] = None
                g.add_node("depthwise_conv2d",
                           [srcs[0], f, self.const(layer, "dw_bias", np.zeros(c * mult))],
                           [mid], attrs, layer.name + "_dw")
                pk = layer.pointwise_kernel.numpy()        # [1, 1, c*m, out]
                pf = self.const(layer, "pw_filter", np.transpose(pk, (3, 0, 1, 2)))
                pattrs = {"stride": (1, 1), "padding": "VALID", "dilation": (1, 1)}
                kind = self.apply_activation(layer, None, out, pattrs)
                y = g.add_tensor(layer.name, _shape(out)).name
                g.add_node("conv2d", [mid, pf, self.bias(layer, pk.shape[3])], [y], pattrs,
                           layer.name)
                return self.finish_activation(layer, kind, y, out)
            kind = self.apply_activation(layer, None, out, attrs)
            y = g.add_tensor(layer.name, _shape(out)).name
            g.add_node("depthwise_conv2d", [srcs[0], f, self.bias(layer, c * mult)], [y],
                       attrs, layer.name)
            return self.finish_activation(layer, kind, y, out)

        if cls == "Dense":
            in_shape = _shape(ins[0])
            if len(in_shape) != 2 and int(np.prod(in_shape[1:-1])) != 1:
                raise NotImplementedError("Dense %s on a rank-%d input (add a Flatten)"
                                          % (layer.name, len(in_shape)))
            x = srcs[0]
            if len(in_shape) != 2:
                flat = g.add_tensor(layer.name + "_flat", (1, in_shape[-1])).name
                g.add_node("reshape", [x], [flat], name=layer.name + "_flat")
                x = flat
            k = layer.kernel.numpy()
            attrs = {}
            kind = self.apply_activation(layer, None, out, attrs)
            y = g.add_tensor(layer.name, (1, k.shape[1])).name
            g.add_node("dense", [x, self.const(layer, "weights", k), self.bias(layer, k.shape[1])],
                       [y], attrs, layer.name)
            y = self.finish_activation(layer, kind, y, out)
            if _shape(out) != (1, k.shape[1]):
                r = g.add_tensor(layer.name + "_reshape", _shape(out)).name
                g.add_node("reshape", [y], [r], name=layer.name + "_reshape")
                y = r
            return y

        if cls == "BatchNormalization":
            axis = cfg["axis"]
            axis = axis[0] if isinstance(axis, (list, tuple)) else axis
            if axis not in (-1, len(_shape(ins[0])) - 1):
                raise NotImplementedError("BatchNormalization %s over a non-channel axis"
                                          % layer.name)
            n = int(layer.moving_mean.shape[0])
            gamma = layer.gamma.numpy() if layer.gamma is not None else np.ones(n)
            beta = layer.beta.numpy() if layer.beta is not None else np.zeros(n)
            y = self.act(out, layer)
            g.add_node("batch_norm", [srcs[0], self.const(layer, "mean", layer.moving_mean.numpy()),
                                      self.const(layer, "variance", layer.moving_variance.numpy()),
                                      self.const(layer, "gamma", gamma),
                                      self.const(layer, "beta", beta)],
                       [y], {"epsilon": float(cfg["epsilon"])}, layer.name)
            return y

        if cls == "ReLU":
            if cfg.get("negative_slope") or cfg.get("threshold"):
                raise NotImplementedError("leaky / thresholded ReLU %s" % layer.name)
            mx = cfg.get("max_value")
            y = self.act(out, layer)
            if mx is None:
                attrs = {"kind": "relu"}
            elif float(mx) == 6.0:
                attrs = {"kind": "relu6"}
            else:
                attrs = {"kind": "clip", "lo": 0.0, "hi": float(mx)}
            g.add_node("activation", [srcs[0]], [y], attrs, layer.name)
            return y

        if cls == "Activation":
            a = _act_name(layer)
            if a == "linear":
                return srcs[0]
            y = self.act(out, layer)
            if a == "softmax":
                g.add_node("softmax", [srcs[0]], [y], {"beta": 1.0}, layer.name)
            elif a in ("relu", "relu6", "sigmoid", "tanh"):
                g.add_node("activation", [srcs[0]], [y], {"kind": a}, layer.name)
            else:
                raise NotImplementedError("activation %r in layer %s" % (a, layer.name))
            return y

        if cls == "Softmax":
            axis = cfg.get("axis", -1)
            axis = axis[0] if isinstance(axis, (list, tuple)) else axis
            if axis not in (-1, len(_shape(ins[0])) - 1):
                raise NotImplementedError("Softmax %s over a non-last axis" % layer.name)
            if int(np.prod(_shape(ins[0])[:-1])) != 1:
                raise NotImplementedError("Softmax %s over more than one row" % layer.name)
            y = self.act(out, layer)
            g.add_node("softmax", [srcs[0]], [y], {"beta": 1.0}, layer.name)
            return y

        if cls in ("MaxPooling2D", "AveragePooling2D"):
            y = self.act(out, layer)
            pool = _pair(cfg["pool_size"])
            stride = _pair(cfg["strides"] or cfg["pool_size"])
            g.add_node("max_pool2d" if cls == "MaxPooling2D" else "avg_pool2d", [srcs[0]], [y],
                       {"kernel": pool, "stride": stride, "padding": cfg["padding"].upper()},
                       layer.name)
            return y

        if cls == "GlobalAveragePooling2D":
            y = self.act(out, layer)
            g.add_node("global_avg_pool", [srcs[0]], [y], name=layer.name)
            return y

        if cls in ("Flatten", "Reshape"):
            if _shape(out) == _shape(ins[0]):
                return srcs[0]
            y = self.act(out, layer)
            g.add_node("reshape", [srcs[0]], [y], name=layer.name)
            return y

        if cls == "Add":
            cur = srcs[0]
            for i, s in enumerate(srcs[1:]):
                y = self.act(out, layer) if i == len(srcs) - 2 else \
                    g.add_tensor("%s_%d" % (layer.name, i), _shape(out)).name
                g.add_node("add", [cur, s], [y], {"activation": None}, layer.name)
                cur = y
            return cur

        raise NotImplementedError("Keras layer %s (%s) has no pTensor mapping yet" % (cls, layer.name))
