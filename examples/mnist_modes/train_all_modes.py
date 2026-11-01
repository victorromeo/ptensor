"""Train MNIST in every pTensor weight storage mode and benchmark each on pTensor.

For every mode this runs the library's QAT flow (current size -> estimated
size -> quantisation-aware fine-tuning -> accuracy and final size), exports
the model as a .ptm file, and runs it with the C++ runner (mnist_modes) over
the MNIST test set to measure accuracy and inference time on pTensor itself.

    # from the repository root
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON \
          -DPTENSOR_MNIST_RUNNER=ON && cmake --build build -j
    pip install tensorflow numpy
    PYTHONPATH=python python3 examples/mnist_modes/train_all_modes.py

Writes examples/mnist_modes/results.json and RESULTS.md (the tables).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples", "mnist"))

import mnist_data  # noqa: E402
from ptensor import qat, ptm  # noqa: E402

FLOAT_MODES = ["float64", "float32", "float16", "int16", "int8",
               "uint7", "uint6", "uint5", "uint4", "uint3", "uint2",
               "hough4", "hough3", "hough2"]

# The first dense layer sees a flattened 7x7x16 map; let the Hough encoder
# also try pairing spatial neighbours by giving it the unflattened view.
VIEW_SHAPES = {"dense": (7, 7, 16, 32)}


def build_model(tf):
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu", name="dense"),
        tf.keras.layers.Dense(10, name="logits"),
        tf.keras.layers.Softmax(),
    ])


def write_test_set(path, x, y):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHH", len(x), 28, 28, 1))
        f.write(np.ascontiguousarray(x, np.uint8).tobytes())
        f.write(np.ascontiguousarray(y, np.uint8).tobytes())


def run_runner(runner, model_path, test_path, ram=False, limit=None):
    cmd = [runner, model_path, test_path]
    if ram:
        cmd.append("--ram")
    if limit:
        cmd += ["--limit", str(limit)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=3, help="float pre-training epochs")
    ap.add_argument("--qat-epochs", type=int, default=2, help="QAT fine-tuning epochs per mode")
    ap.add_argument("--modes", default=",".join(FLOAT_MODES + ["int8-full"]))
    ap.add_argument("--runner", default=os.path.join(REPO, "build", "examples", "mnist_modes", "mnist_modes"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--limit", type=int, default=None, help="test images for the C++ runs")
    ap.add_argument("--seed", type=int, default=0,
                    help="QAT shuffling seed; a nonzero seed writes results_seed<N>.json only")
    ap.add_argument("--bench-only", action="store_true",
                    help="re-time the .ptm files from the last run (no training)")
    args = ap.parse_args()
    if args.bench_only:
        return bench_only(args)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = args.out_dir if not args.seed else os.path.join(args.out_dir, "seed%d" % args.seed)
    os.makedirs(model_dir, exist_ok=True)

    x_train, y_train = mnist_data.load("train")
    x_test, y_test = mnist_data.load("test")
    test_path = os.path.join(args.out_dir, "mnist_test.bin")
    write_test_set(test_path, x_test, y_test)
    xf_train = (x_train.astype(np.float32) / 255.0)[..., None]
    xf_test = (x_test.astype(np.float32) / 255.0)[..., None]
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Float baseline, trained once; every mode fine-tunes from it.
    base_path = os.path.join(args.out_dir, "baseline.weights.h5")
    base = build_model(tf)
    if os.path.exists(base_path):
        base.load_weights(base_path)
    else:
        base.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        base.fit(xf_train, y_train, epochs=args.epochs, batch_size=128, verbose=2)
        base.save_weights(base_path)
    base_acc = float((base.predict(xf_test, batch_size=512, verbose=0).argmax(-1) == y_test).mean())
    print("float baseline (Keras): %.2f%%" % (100 * base_acc))

    results = []
    for mode in args.modes.split(","):
        print("\n=== %s ===" % mode)
        t0 = time.time()
        tf.keras.utils.set_random_seed(args.seed)
        model = build_model(tf)
        model.set_weights(base.get_weights())
        path = os.path.join(model_dir, "mnist_%s.ptm" % mode)
        row = {"mode": mode}
        if mode == "int8-full":
            # Full-integer int8 quantises activations too, so it runs on the
            # TFLM int8 kernels. Give it the same fine-tuning budget in float,
            # then post-training quantise it the way TFLite does.
            q = qat.QATModel(model, "float32")
            row["float32_bytes"] = q.report_current()
            q.fit(xf_train, y_train, epochs=args.qat_epochs, seed=args.seed)
            file_bytes, weight_bytes = ptm.write_int8_model(path, model, xf_train[:2000])
            row.update(ptq_accuracy=None, keras_accuracy=None, bytes=weight_bytes,
                       estimated_bytes=weight_bytes, storage="int8 weights + int8 activations")
        else:
            q = qat.QATModel(model, mode, view_shapes=VIEW_SHAPES)
            row["float32_bytes"] = q.report_current()
            row["estimated_bytes"] = q.report_estimate()
            row["ptq_accuracy"] = q.evaluate(xf_test, y_test)
            print("      before training (post-training quantisation): %.2f%%" % (100 * row["ptq_accuracy"]))
            q.fit(xf_train, y_train, epochs=args.qat_epochs, seed=args.seed)
            acc, size = q.report_final(xf_test, y_test)
            row.update(keras_accuracy=acc, bytes=size,
                       storage=", ".join("%s: %s" % (l.name, e.describe()) for l, e, _ in q.export()))
            ptm.write_float_model(path, model, q.export())
        row["file_bytes"] = os.path.getsize(path)
        stream = run_runner(args.runner, path, test_path, limit=args.limit)
        row["accuracy"] = stream["accuracy"]
        row["us_per_image"] = stream["us_per_image"]
        row["runner_weight_bytes"] = stream["weight_bytes"]
        if mode not in ("float32", "int8-full"):
            ram = run_runner(args.runner, path, test_path, ram=True, limit=args.limit)
            row["us_per_image_ram"] = ram["us_per_image"]
            assert abs(ram["accuracy"] - stream["accuracy"]) < 1e-9, "decode-to-RAM changed results"
        print("pTensor C++: accuracy %.2f%%, %.1f us/image, weights %d B, train+export %.0fs"
              % (100 * row["accuracy"], row["us_per_image"], row["runner_weight_bytes"], time.time() - t0))
        results.append(row)

    if args.seed:
        with open(os.path.join(HERE, "results_seed%d.json" % args.seed), "w") as f:
            json.dump({"seed": args.seed, "results": results}, f, indent=1)
        return
    # Running a subset of modes updates just those rows of the previous run.
    order = FLOAT_MODES + ["int8-full"]
    merged = {r["mode"]: r for r in _load_results()}
    merged.update({r["mode"]: r for r in results})
    results = sorted(merged.values(), key=lambda r: order.index(r["mode"]) if r["mode"] in order else 99)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"baseline_keras_accuracy": base_acc, "results": results}, f, indent=1)
    write_markdown(os.path.join(HERE, "RESULTS.md"), base_acc, results)


def _load_results():
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["results"]


def bench_only(args):
    with open(os.path.join(HERE, "results.json")) as f:
        data = json.load(f)
    test_path = os.path.join(args.out_dir, "mnist_test.bin")
    for row in data["results"]:
        path = os.path.join(args.out_dir, "mnist_%s.ptm" % row["mode"])
        stream = run_runner(args.runner, path, test_path, limit=args.limit)
        assert abs(stream["accuracy"] - row["accuracy"]) < 1e-9, row["mode"]
        row["us_per_image"] = stream["us_per_image"]
        if "us_per_image_ram" in row:
            row["us_per_image_ram"] = run_runner(args.runner, path, test_path, ram=True,
                                                 limit=args.limit)["us_per_image"]
        print("%-10s %.1f us/image" % (row["mode"], row["us_per_image"]))
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(data, f, indent=1)
    write_markdown(os.path.join(HERE, "RESULTS.md"), data["baseline_keras_accuracy"], data["results"])


def write_markdown(path, base_acc, results):
    f32 = next((r for r in results if r["mode"] == "float32"), results[0])
    lines = [
        "# MNIST in every pTensor weight storage mode",
        "",
        "Generated by `train_all_modes.py`. Network: Conv 3x3x8 → MaxPool → Conv 3x3x16 → MaxPool →",
        "Dense 32 → Dense 10 (26,882 parameters). Float baseline trained 3 epochs, then every mode",
        "fine-tuned 2 epochs with quantisation-aware training from that baseline (float32 too, so all",
        "rows have the same training budget). Accuracy and time are measured by the C++ runner on",
        "pTensor over the full 10,000-image test set, batch 1, one x86-64 core (Release build);",
        "time excludes softmax (prediction is the argmax of the logits).",
        "",
        "Float baseline before fine-tuning (Keras): %.2f%%." % (100 * base_acc),
        "",
        "| Mode | Weights stored as | Size | vs float32 | Estimated | Accuracy before QAT | Accuracy (pTensor) | µs / image (decode on the fly) | µs / image (decoded to RAM once) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        ptq = "%.2f%%" % (100 * r["ptq_accuracy"]) if r.get("ptq_accuracy") is not None else "n/a"
        ram = "%.0f" % r["us_per_image_ram"] if "us_per_image_ram" in r else "same"
        lines.append("| %s | %s | %d B | %.1fx | %d B | %s | %.2f%% | %.0f | %s |" % (
            r["mode"], _short_storage(r), r["runner_weight_bytes"],
            f32["runner_weight_bytes"] / float(r["runner_weight_bytes"]),
            r["estimated_bytes"], ptq, 100 * r["accuracy"], r["us_per_image"], ram))
    lines += [
        "",
        "Decode on the fly keeps the weights compressed in flash and decodes them as operators read",
        "them, so it uses no extra RAM; operators fall back to their reference loops for non-dense",
        "tensors, which is why those columns are ~25x slower than float32 on this host. Decoding once",
        "to RAM at start-up restores the dense fast paths (same speed as float32) at the cost of",
        "107 KB of RAM for these weights. Both give identical predictions.",
    ]
    seeds = _seed_rows(results)
    if seeds:
        lines += ["", "## Repeat runs", "",
                  "Accuracy (pTensor, 10,000 images) with different QAT shuffling seeds, same float baseline:", "",
                  "| Mode | Size | " + " | ".join("seed %d" % k for k in sorted(seeds[0][1])) + " | Mean |",
                  "| --- | ---: | " + " | ".join("---:" for _ in seeds[0][1]) + " | ---: |"]
        for mode, accs, size in seeds:
            vals = [accs[k] for k in sorted(accs)]
            lines.append("| %s | %d B | %s | %.2f%% |" % (
                mode, size, " | ".join("%.2f%%" % (100 * v) for v in vals), 100 * np.mean(vals)))
    lines += ["", "Per-layer storage:", ""]
    for r in results:
        lines.append("- **%s**: %s" % (r["mode"], r["storage"]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _seed_rows(results):
    import glob
    runs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results_seed*.json"))):
        with open(path) as f:
            d = json.load(f)
        for r in d["results"]:
            runs.setdefault(r["mode"], {})[d["seed"]] = r["accuracy"]
    rows = []
    for r in results:
        if r["mode"] in runs:
            accs = dict(runs[r["mode"]])
            accs[0] = r["accuracy"]
            rows.append((r["mode"], accs, r["runner_weight_bytes"]))
    return rows


def _short_storage(r):
    m = r["mode"]
    if m == "int8-full":
        return "int8 (per-channel), int8 activations"
    if m.startswith("hough"):
        return "HoughRomTensor + uint fallback"
    if m == "float32":
        return "RomTensor float32"
    return "PackedRomTensor " + m


if __name__ == "__main__":
    main()
