"""TensorFlow SavedModel directories and concrete functions.

TensorFlow graphs have hundreds of op types, so instead of a second importer
this frontend runs TensorFlow's own converter to a float TFLite flatbuffer
(which folds constants, fuses bias / activations and drops training-only
ops) and imports that with the TFLite frontend.
"""
import os

from .base import Frontend, FrontendUnavailable, register_frontend


@register_frontend
class TFSavedModelFrontend(Frontend):
    name = "tf"

    def accepts(self, source):
        if isinstance(source, str):
            return os.path.isdir(source) and os.path.exists(os.path.join(source, "saved_model.pb"))
        mod = type(source).__module__
        return "tensorflow" in mod and hasattr(source, "graph") and hasattr(source, "inputs")

    def load(self, source, name=None, **options):
        try:
            import tensorflow as tf
        except ImportError:
            raise FrontendUnavailable("the TensorFlow frontend needs tensorflow installed")
        if isinstance(source, str):
            conv = tf.lite.TFLiteConverter.from_saved_model(source)
            name = name or os.path.basename(os.path.normpath(source))
        else:
            conv = tf.lite.TFLiteConverter.from_concrete_functions([source])
        flatbuffer = conv.convert()
        from .base import get_frontend
        graph = get_frontend("tflite").load(flatbuffer, name=name or "model")
        graph.source = "tf"
        graph.notes.append("converted with tf.lite.TFLiteConverter (float32), then imported as TFLite")
        return graph
