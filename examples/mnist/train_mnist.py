"""Train a small MNIST CNN and export it as a full-integer int8 TFLite model.

This is how mnist_int8.tflite was made. You only need it to retrain; the
sample (mnist_ptensor.py) runs on the committed model without TensorFlow.

    pip install tensorflow
    python3 train_mnist.py            # writes mnist_int8.tflite (and mnist_f32.tflite)

The int8 model is what a microcontroller would run: int8 weights (per-channel
for conv, per-tensor for dense), int8 activations, int32 biases, and int8
input/output tensors.
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import mnist_data

HERE = os.path.dirname(os.path.abspath(__file__))


def build_model(batch_size=None):
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1), batch_size=batch_size),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(0)
    x_train, y_train = mnist_data.load("train")
    x_test, y_test = mnist_data.load("test")
    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=128,
              validation_split=0.1, verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print("float Keras test accuracy: %.4f" % acc)

    # Export with a fixed batch of 1, as a device runs it. This also keeps
    # Flatten a static RESHAPE instead of a SHAPE/STRIDED_SLICE/PACK chain.
    infer = build_model(batch_size=1)
    infer.set_weights(model.get_weights())
    model = infer

    # Float32 model, for comparison.
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(os.path.join(args.out_dir, "mnist_f32.tflite"), "wb") as f:
        f.write(conv.convert())

    # Full-integer int8 model.
    def representative():
        for i in range(0, 2000, 4):
            yield [x_train[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    path = os.path.join(args.out_dir, "mnist_int8.tflite")
    with open(path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (path, len(blob)))


if __name__ == "__main__":
    main()
