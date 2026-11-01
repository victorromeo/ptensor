"""Structured pruning of the MNIST CNN, alone and combined with quantisation.

Starts from the same cached float baseline as examples/mnist_modes, prunes
whole filters/units with ptensor.prune.PrunedModel (report current size ->
estimated size -> prune + fine-tune -> accuracy and final size), optionally
quantises the smaller model with QATModel, writes a .ptm file and runs it on
pTensor with the mnist_modes C++ runner to measure accuracy, size and time.

    # from the repository root
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON \
          -DPTENSOR_MNIST_RUNNER=ON && cmake --build build -j
    pip install tensorflow numpy
    PYTHONPATH=python python3 examples/mnist_pruning/prune_mnist.py

Writes examples/mnist_pruning/results.json and RESULTS.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples", "mnist"))
sys.path.insert(0, os.path.join(REPO, "examples", "mnist_modes"))

import mnist_data  # noqa: E402
import train_all_modes as tam  # noqa: E402  (same network, baseline and runner)
from ptensor import prune, ptm, qat  # noqa: E402

# name -> (ratio, pruning steps, storage mode after pruning or None)
RUNS = [
    ("float32", 0.0, 1, None),
    ("prune25", 0.25, 1, None),
    ("prune50", 0.5, 1, None),
    ("prune75", 0.75, 1, None),
    ("prune75-2step", 0.75, 2, None),
    ("prune90", 0.9, 1, None),
    ("prune90-2step", 0.9, 2, None),
    ("prune50+int8", 0.5, 1, "int8"),
    ("prune50+uint4", 0.5, 1, "uint4"),
    ("prune75+uint4", 0.75, 1, "uint4"),
    ("prune50+hough2", 0.5, 1, "hough2"),
    ("prune50+int8-full", 0.5, 1, "int8-full"),
    ("narrow50", 0.5, 0, None),
    ("narrow75", 0.75, 0, None),
]
# Unpruned rows from examples/mnist_modes/results.json, for comparison.
REFERENCE_MODES = ("int8", "uint4", "hough2", "int8-full")


def build_narrow(tf, ratio):
    """The baseline architecture with every hidden layer `ratio` narrower,
    trained from scratch: does pruning beat simply designing a smaller net?"""
    L = tf.keras.layers
    n = lambda c: prune.n_keep(c, ratio)  # noqa: E731
    return tf.keras.Sequential([
        L.Input((28, 28, 1)),
        L.Conv2D(n(8), 3, padding="same", activation="relu", name="conv1"),
        L.MaxPooling2D(2),
        L.Conv2D(n(16), 3, padding="same", activation="relu", name="conv2"),
        L.MaxPooling2D(2),
        L.Flatten(),
        L.Dense(n(32), activation="relu", name="dense"),
        L.Dense(10, name="logits"),
        L.Softmax(),
    ])


def keras_acc(model, x, y):
    return float((model.predict(x, batch_size=512, verbose=0).argmax(-1) == y).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=2,
                    help="fine-tuning epochs after pruning (and QAT epochs)")
    ap.add_argument("--runs", default=",".join(r[0] for r in RUNS))
    ap.add_argument("--runner", default=os.path.join(REPO, "build", "examples", "mnist_modes", "mnist_modes"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "examples", "mnist_modes", "out"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    x_train, y_train = mnist_data.load("train")
    x_test, y_test = mnist_data.load("test")
    test_path = os.path.join(args.out_dir, "mnist_test.bin")
    if not os.path.exists(test_path):
        tam.write_test_set(test_path, x_test, y_test)
    xf_train = (x_train.astype(np.float32) / 255.0)[..., None]
    xf_test = (x_test.astype(np.float32) / 255.0)[..., None]
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    base_path = os.path.join(args.out_dir, "baseline.weights.h5")
    base = tam.build_model(tf)
    if os.path.exists(base_path):
        base.load_weights(base_path)
    else:  # same recipe as train_all_modes.py
        base.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        base.fit(xf_train, y_train, epochs=3, batch_size=128, verbose=2)
        base.save_weights(base_path)
    base_acc = keras_acc(base, xf_test, y_test)
    print("float baseline (Keras): %.2f%%" % (100 * base_acc))

    print("\n=== sensitivity: one layer pruned at a time, no fine-tuning ===")
    sens = prune.sensitivity(base, xf_test, y_test, ratios=(0.25, 0.5, 0.75, 0.9))

    wanted = args.runs.split(",")
    results = []
    for name, ratio, steps, mode in RUNS:
        if name not in wanted:
            continue
        print("\n=== %s ===" % name)
        t0 = time.time()
        tf.keras.utils.set_random_seed(0)
        row = {"run": name, "ratio": ratio, "steps": steps, "mode": mode or "float32"}
        if steps == 0:  # narrow network trained from scratch, same total epochs as baseline + fine-tune
            model = build_narrow(tf, ratio)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(xf_train, y_train, epochs=3 + args.epochs, batch_size=128, verbose=0)
            row["oneshot_accuracy"] = None
        else:
            model = tam.build_model(tf)
            model.set_weights(base.get_weights())
            p = prune.PrunedModel(model, ratio)
            p.report_current()
            p.report_estimate()
            row["oneshot_accuracy"] = keras_acc(p.estimated_model(), xf_test, y_test)
            print("      one-shot, before fine-tuning: %.2f%%" % (100 * row["oneshot_accuracy"]))
            if mode is None:
                # `steps` rounds sharing the fine-tuning budget
                p.fit(xf_train, y_train, epochs=max(1, args.epochs // steps), steps=steps)
            else:
                # prune only; QAT below does the fine-tuning (same budget as the unpruned modes)
                p.fit(xf_train, y_train, epochs=0, steps=steps)
            p.report_final(xf_test, y_test)
            model = p.model
        row["widths"] = [l.get_weights()[0].shape[-1] for l in model.layers if hasattr(l, "kernel")]
        row["params"], row["float32_bytes"], row["macs"] = prune.model_stats(model)
        path = os.path.join(args.out_dir, "mnist_%s.ptm" % name.replace("+", "_"))
        if mode == "int8-full":
            q = qat.QATModel(model, "float32")
            q.fit(xf_train, y_train, epochs=args.epochs)
            ptm.write_int8_model(path, model, xf_train[:2000])
        elif mode is not None:
            q = qat.QATModel(model, mode, view_shapes=_views(model))
            q.report_current()
            q.report_estimate()
            q.fit(xf_train, y_train, epochs=args.epochs)
            q.report_final(xf_test, y_test)
            ptm.write_float_model(path, model, q.export())
        else:
            q = qat.QATModel(model, "float32")
            ptm.write_float_model(path, model, q.export())
        out = tam.run_runner(args.runner, path, test_path, limit=args.limit)
        row["accuracy"] = out["accuracy"]
        row["us_per_image"] = out["us_per_image"]
        row["bytes"] = out["weight_bytes"]
        if mode not in (None, "int8-full"):
            row["us_per_image_ram"] = tam.run_runner(args.runner, path, test_path, ram=True,
                                                     limit=args.limit)["us_per_image"]
        print("pTensor C++: accuracy %.2f%%, %.1f us/image, weights %d B, %.0fs"
              % (100 * row["accuracy"], row["us_per_image"], row["bytes"], time.time() - t0))
        results.append(row)

    old = {}
    rpath = os.path.join(HERE, "results.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            old = {r["run"]: r for r in json.load(f)["results"]}
    old.update({r["run"]: r for r in results})
    order = [r[0] for r in RUNS]
    results = sorted(old.values(), key=lambda r: order.index(r["run"]) if r["run"] in order else 99)
    with open(rpath, "w") as f:
        json.dump({"baseline_keras_accuracy": base_acc, "sensitivity": sens, "results": results}, f, indent=1)
    write_markdown(os.path.join(HERE, "RESULTS.md"), base_acc, sens, results)


def _views(model):
    # Hough pairing view for the first dense layer: the unflattened map.
    for layer in model.layers:
        if type(layer).__name__ == "Flatten":
            h, w, c = (int(d) for d in layer.input.shape[1:])
            units = model.get_layer("dense").kernel.shape[-1]
            return {"dense": (h, w, c, units)}
    return None


def _reference():
    path = os.path.join(REPO, "examples", "mnist_modes", "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["mode"]: r for r in json.load(f)["results"]}


def write_markdown(path, base_acc, sens, results):
    f32 = next((r for r in results if r["run"] == "float32"), results[0])
    ref = _reference()
    lines = [
        "# Structured pruning on MNIST",
        "",
        "Generated by `prune_mnist.py`. Same network and float baseline as",
        "[mnist_modes](../mnist_modes/RESULTS.md): Conv 3x3x8 → MaxPool → Conv 3x3x16 → MaxPool →",
        "Dense 32 → Dense 10, trained 3 epochs (Keras %.2f%%). Each run removes the given fraction" % (100 * base_acc),
        "of filters/units from conv1, conv2 and dense (lowest L1 norm first; the logits layer is",
        "kept), then fine-tunes 2 epochs, so every row has the same training budget as the",
        "mnist_modes rows. `2step` prunes in two rounds with one epoch after each. `+mode` rows",
        "prune, then fine-tune with 2 epochs of QAT in that storage mode instead. `narrow` rows",
        "train the same smaller widths from scratch for 5 epochs (3 + 2). Accuracy and time are",
        "measured by the mnist_modes C++ runner on pTensor (10,000 test images, batch 1, one",
        "x86-64 core, Release build).",
        "",
        "| Run | Widths | Weights | Size | vs float32 | MACs | One-shot accuracy | Accuracy (pTensor) | µs / image |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        one = "%.2f%%" % (100 * r["oneshot_accuracy"]) if r.get("oneshot_accuracy") is not None else "n/a"
        us = "%.0f" % r["us_per_image"]
        if "us_per_image_ram" in r:
            us += " (%.0f in RAM)" % r["us_per_image_ram"]
        lines.append("| %s | %s | %d | %d B | %.1fx | %d | %s | %.2f%% | %s |" % (
            r["run"], "-".join(str(w) for w in r["widths"]), r["params"], r["bytes"],
            f32["bytes"] / float(r["bytes"]), r["macs"], one, 100 * r["accuracy"], us))
    rows = [m for m in REFERENCE_MODES if m in ref]
    if rows:
        lines += ["", "Unpruned rows from mnist_modes for comparison (same 2-epoch QAT budget):", "",
                  "| Mode | Size | Accuracy (pTensor) | µs / image |", "| --- | ---: | ---: | ---: |"]
        for m in rows:
            r = ref[m]
            us = "%.0f" % r["us_per_image"]
            if "us_per_image_ram" in r:
                us += " (%.0f in RAM)" % r["us_per_image_ram"]
            lines.append("| %s | %d B | %.2f%% | %s |" % (m, r["runner_weight_bytes"], 100 * r["accuracy"], us))
    lines += ["", "## Sensitivity", "",
              "Keras accuracy with one layer pruned on its own, no fine-tuning:", "",
              "| Layer | " + " | ".join("%d%% removed" % (100 * float(k)) for k in next(iter(sens.values()))) + " |",
              "| --- | " + " | ".join("---:" for _ in next(iter(sens.values()))) + " |"]
    for layer, accs in sens.items():
        lines.append("| %s | %s |" % (layer, " | ".join("%.2f%%" % (100 * a) for a in accs.values())))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
