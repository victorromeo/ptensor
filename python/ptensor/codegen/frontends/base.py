"""Frontend interface and registry."""
import os

_REGISTRY = []


class FrontendUnavailable(NotImplementedError):
    """The frontend exists but cannot import this model (missing package, or
    the importer is not written yet)."""


class Frontend(object):
    """Turns a model of one framework into an ``ir.Graph``.

    Subclasses set ``name`` and ``extensions`` and implement ``accepts`` (is
    this object / path one of mine?) and ``load``. ``load`` must return a
    graph that follows the conventions in ``ir`` (NHWC, OHWI filters, dense
    weights [in, out], a bias on every linear layer, batch 1).
    """
    name = None
    extensions = ()

    def accepts(self, source):
        return isinstance(source, str) and source.lower().endswith(tuple(self.extensions))

    def load(self, source, **options):
        raise NotImplementedError

    def __repr__(self):
        return "<frontend %s>" % self.name


def register_frontend(cls):
    """Class decorator: make a Frontend available to load_model()."""
    inst = cls() if isinstance(cls, type) else cls
    for i, f in enumerate(_REGISTRY):
        if f.name == inst.name:
            _REGISTRY[i] = inst
            return cls
    _REGISTRY.append(inst)
    return cls


def available_frontends():
    return [f.name for f in _REGISTRY]


def get_frontend(name):
    for f in _REGISTRY:
        if f.name == name:
            return f
    raise KeyError("no frontend %r (have: %s)" % (name, ", ".join(available_frontends())))


def load_model(source, framework=None, **options):
    """Import a model (object or path) into an ir.Graph.

    framework: a frontend name ("keras", "tflite", "tf", "pytorch", "onnx"),
    or None to pick the first frontend that accepts the source.
    """
    from ..ir import Graph
    if isinstance(source, Graph):
        return source
    if framework is not None:
        frontend = get_frontend(framework)
    else:
        if isinstance(source, str) and not os.path.exists(source):
            raise FileNotFoundError(source)
        matches = [f for f in _REGISTRY if f.accepts(source)]
        if not matches:
            raise ValueError("no frontend recognises %r; pass framework= (have: %s)"
                             % (source if isinstance(source, str) else type(source),
                                ", ".join(available_frontends())))
        frontend = matches[0]
    graph = frontend.load(source, **options)
    graph.source = graph.source or frontend.name
    return graph.validate()
