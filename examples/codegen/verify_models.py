"""Generate pTensor C++ for MNIST models in many modes, compile each one, run it
on the test set and check it against Python.

For every mode the script runs ptensor.codegen.CodeGenerator, writes a small
harness, compiles harness + generated model against libptensor, feeds it the
test images and compares the outputs with

  * float models: the NumPy reference interpreter on the generated graph
    (weights decoded exactly as the device decodes them), and the original
    Keras / TFLite model;
  * int8 models: pTensor's own kernels through the Python bindings (must be
    bit-exact) and, for .tflite inputs, TFLite's reference kernels.

    # from the repository root
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTENSOR_BUILD_C_API=ON && cmake --build build -j
    pip install tensorflow tflite numpy
    PYTHONPATH=python python3 examples/codegen/verify_models.py --out /tmp/cg

--modes a,b to run a subset, -n to change the number of test images,
--cmake-check to also build two modes through their generated .cmake file
with pTensor's unused features switched off.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples", "mnist"))

import mnist_data  # noqa: E402
from ptensor.codegen import CodeGenerator, reference  # noqa: E402

HARNESS = r"""
#include <chrono>
#include <cstdio>
#include <vector>
#include "%(name)s.h"
typedef %(in_t)s in_t;
typedef %(out_t)s out_t;
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  FILE* g = fopen(argv[2], "wb");
  if (!f || !g) return 3;
  const size_t n_in = %(N)s_IN0_SIZE, n_out = %(N)s_OUT0_SIZE;
  std::vector<in_t> x(n_in);
  std::vector<out_t> y(n_out);
  double us = 0;
  unsigned count = 0;
  while (fread(x.data(), sizeof(in_t), n_in, f) == n_in) {
    auto t0 = std::chrono::steady_clock::now();
    int rc = %(name)s_invoke(x.data(), y.data());
    us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (rc) { fprintf(stderr, "invoke failed (%%d) on example %%u\n", rc, count); return 1; }
    fwrite(y.data(), sizeof(out_t), n_out, g);
    ++count;
  }
  printf("{\"count\": %%u, \"us\": %%.2f, \"meta_hw\": %%zu, \"ram_hw\": %%zu}\n", count,
         count ? us / count : 0.0, %(name)s_meta_high_water(), %(name)s_ram_high_water());
  return 0;
}
"""

CTYPES = {"float32": ("float", np.float32), "int8": ("int8_t", np.int8)}


def compile_and_run(gen, out_dir, build, x_dev):
    paths = gen.generate(out_dir)
    g = gen.graph
    tin, tout = g.tensors[g.inputs[0]], g.tensors[g.outputs[0]]
    with open(os.path.join(out_dir, "main.cpp"), "w") as f:
        f.write(HARNESS % {"name": gen.name, "N": gen.name.upper(),
                           "in_t": CTYPES[tin.dtype][0], "out_t": CTYPES[tout.dtype][0]})
    exe = os.path.join(out_dir, "run")
    inc = []
    for d in ("src", "src/pTensor", "src/pTensor/core", "src/pTensor/allocators",
              "src/pTensor/tensors", "src/pTensor/ops", "src/pTensor/ops/reference",
              "src/pTensor/ops/tflm", "src/pTensor/errorHandlers"):
        inc += ["-I", os.path.join(REPO, d)]
    cmd = ["g++", "-std=c++11", "-O2", "-fno-rtti", "-Wall"] + inc + [
           "-I", os.path.join(build, "src", "pTensor", "core", "generated"),
           os.path.join(out_dir, gen.name + ".cpp"), os.path.join(out_dir, "main.cpp"),
           os.path.join(build, "src", "pTensor", "libptensor.a"),
           os.path.join(build, "src", "pTensor", "core", "libptensor_core.a"), "-o", exe]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("compile failed:\n" + r.stderr[-4000:])
    compile_s = time.time() - t0
    x_dev.astype(CTYPES[tin.dtype][1]).tofile(os.path.join(out_dir, "in.bin"))
    r = subprocess.run([exe, os.path.join(out_dir, "in.bin"), os.path.join(out_dir, "out.bin")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("run failed: " + r.stderr[-2000:])
    stats = json.loads(r.stdout)
    y = np.fromfile(os.path.join(out_dir, "out.bin"), CTYPES[tout.dtype][1]).reshape(len(x_dev), -1)
    stats["compile_s"] = round(compile_s, 1)
    return y, stats


CMAKE_PROJECT = """cmake_minimum_required(VERSION 3.10)
project(codegen_check CXX)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
include(${CMAKE_CURRENT_LIST_DIR}/%(name)s.cmake)
add_subdirectory(%(repo)s ptensor)
add_executable(run main.cpp)
target_link_libraries(run %(name)s)
"""


def cmake_build_and_run(gen, out_dir, x_dev):
    """Build the generated model the way a firmware project would: include
    its .cmake (which switches off the pTensor features it does not use),
    add pTensor as a subdirectory, link. Returns outputs, stats and the size
    of the linked program's code."""
    with open(os.path.join(out_dir, "CMakeLists.txt"), "w") as f:
        f.write(CMAKE_PROJECT % {"name": gen.name, "repo": REPO})
    bdir = os.path.join(out_dir, "cmake-build")
    for cmd in (["cmake", "-S", out_dir, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release"],
                ["cmake", "--build", bdir, "-j8"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(" ".join(cmd) + " failed:\n" + (r.stdout + r.stderr)[-4000:])
    cfg = open(os.path.join(bdir, "ptensor", "src", "pTensor", "core", "generated", "pTensor",
                            "ptensor_config.h")).read()
    on = [ln.split()[1].replace("PTENSOR_WITH_", "") for ln in cfg.splitlines()
          if ln.startswith("#define PTENSOR_WITH_") and ln.strip().endswith(" 1")]
    exe = os.path.join(bdir, "run")
    r = subprocess.run([exe, os.path.join(out_dir, "in.bin"), os.path.join(out_dir, "out_cmake.bin")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("run failed: " + r.stderr[-2000:])
    st = json.loads(r.stdout)
    text = int(subprocess.run(["size", exe], capture_output=True, text=True).stdout.splitlines()[1].split()[0])
    return st, on, text


def quantize_input(g, x):
    t = g.tensors[g.inputs[0]]
    if t.dtype != "int8":
        return x
    s, z = float(t.quant.scale[0]), int(t.quant.zero_point[0])
    return np.clip(np.rint(x / s) + z, -128, 127).astype(np.int8)


def tflite_reference(model_path_or_bytes, x):
    import tensorflow as tf
    kw = {"model_content": model_path_or_bytes} if isinstance(model_path_or_bytes, bytes) \
        else {"model_path": model_path_or_bytes}
    it = tf.lite.Interpreter(experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF, **kw)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for xi in x:
        it.set_tensor(i["index"], xi.reshape(i["shape"]).astype(i["dtype"]))
        it.invoke()
        out.append(it.get_tensor(o["index"]).reshape(-1))
    return np.stack(out)


def train_models(work):
    """MNIST CNN (same as examples/mnist) plus a small int8 residual net."""
    import tensorflow as tf
    path = os.path.join(work, "mnist.keras")
    xtr, ytr = mnist_data.load("train")
    xtr = (xtr / 255.0).astype(np.float32)[..., None]
    L = tf.keras.layers
    if not os.path.exists(path):
        tf.keras.utils.set_random_seed(0)
        m = tf.keras.Sequential([
            L.Input((28, 28, 1)), L.Conv2D(8, 3, padding="same", activation="relu"),
            L.MaxPooling2D(2), L.Conv2D(16, 3, padding="same", activation="relu"),
            L.MaxPooling2D(2), L.Flatten(), L.Dense(32, activation="relu"), L.Dense(10),
            L.Softmax()], name="mnist")
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(xtr, ytr, epochs=2, batch_size=128, verbose=0)
        m.save(path)
    rpath = os.path.join(work, "resnet.keras")
    if not os.path.exists(rpath):
        tf.keras.utils.set_random_seed(1)
        i = L.Input((28, 28, 1))
        a = L.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(i)
        a = L.BatchNormalization()(a)
        a = L.ReLU(6.0)(a)
        b = L.DepthwiseConv2D(3, padding="same")(a)
        b = L.BatchNormalization()(b)
        b = L.ReLU()(b)
        c = L.ReLU()(L.Add()([a, b]))
        c = L.MaxPooling2D(2)(c)
        d = L.SeparableConv2D(16, 3, padding="valid", activation="relu6")(c)
        d = L.Flatten()(d)
        d = L.Dense(10)(d)
        e = L.Softmax()(d)
        m = tf.keras.Model(i, e, name="resnet")
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(xtr, ytr, epochs=1, batch_size=128, verbose=0)
        m.save(rpath)
    for name in ("mnist", "resnet"):
        tfl = os.path.join(work, name + "_int8.tflite")
        if not os.path.exists(tfl):
            km = tf.keras.models.load_model(os.path.join(work, name + ".keras"))
            conv = tf.lite.TFLiteConverter.from_keras_model(km)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: ([xtr[k:k + 1]] for k in range(300))
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            open(tfl, "wb").write(conv.convert())
    sm = os.path.join(work, "mnist_savedmodel")
    if not os.path.exists(sm):
        tf.keras.models.load_model(path).export(sm)
    return xtr


def modes(work, calib):
    keras_mnist = os.path.join(work, "mnist.keras")
    resnet = os.path.join(work, "resnet.keras")
    m = [
        ("f32-unfused-dynamic", keras_mnist, dict(fuse=False, memory="dynamic")),
        ("f32-fused-planned", keras_mnist, dict()),
        ("f32-fused-nofast", keras_mnist, dict(fast_kernels=False)),
        ("float16", keras_mnist, dict(weights="float16")),
        ("int8w", keras_mnist, dict(weights="int8")),
        ("uint4", keras_mnist, dict(weights="uint4")),
        ("uint4-ram", keras_mnist, dict(weights="uint4", weights_in_ram=True)),
        ("hough4", keras_mnist, dict(weights="hough4")),
        ("hough4-ram", keras_mnist, dict(weights="hough4", weights_in_ram=True)),
        ("cubic4", keras_mnist, dict(weights="cubic4")),
        ("quad3-unfused", keras_mnist, dict(weights="quad3", fuse=False)),
        ("prune50", keras_mnist, dict(prune=0.5)),
        ("prune50-uint6", keras_mnist, dict(prune=0.5, weights="uint6")),
        ("int8-ptq", keras_mnist, dict(activations="int8", calibration_data=calib,
                                       input_scale=1 / 255.0, input_zero_point=-128)),
        ("int8-ptq-floatio-dyn", keras_mnist, dict(activations="int8", calibration_data=calib,
                                                   float_io=True, memory="dynamic")),
        ("prune25-int8-ptq", keras_mnist, dict(prune=0.25, activations="int8",
                                               calibration_data=calib)),
        ("tflite-int8-repo", os.path.join(REPO, "examples", "mnist", "mnist_int8.tflite"), dict()),
        ("tflite-f32-repo", os.path.join(REPO, "examples", "mnist", "mnist_f32.tflite"),
         dict(fuse=False)),
        ("tflite-int8", os.path.join(work, "mnist_int8.tflite"), dict()),
        ("savedmodel", os.path.join(work, "mnist_savedmodel"), dict()),
        ("resnet-fused", resnet, dict()),
        ("resnet-unfused-dyn", resnet, dict(fuse=False, memory="dynamic")),
        ("resnet-hough4", resnet, dict(weights="hough4", encode_min_size=16)),
        ("resnet-tflite-int8", os.path.join(work, "resnet_int8.tflite"), dict()),
    ]
    return m


# Modes also built through their generated .cmake with --cmake-check.
CMAKE_MODES = ("f32-unfused-dynamic", "f32-fused-planned", "f32-fused-nofast", "hough4-ram",
               "cubic4", "tflite-int8-repo", "resnet-tflite-int8")


def original_output(src, x):
    """Output of the model the generator started from."""
    if src.endswith(".keras"):
        import keras
        return keras.models.load_model(src).predict(x, verbose=0, batch_size=256)
    if src.endswith(".tflite"):
        return None  # compared through tflite_reference instead
    import tensorflow as tf
    loaded = tf.saved_model.load(src)
    return loaded.serve(x).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ptensor_codegen")
    ap.add_argument("--build", default=os.path.join(REPO, "build"))
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--modes")
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--cmake-check", action="store_true",
                    help="also build some modes through their generated .cmake")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    xtr = train_models(a.out)
    calib = xtr[:500]
    xte, yte = mnist_data.load("test")
    xte = (xte[:a.n] / 255.0).astype(np.float32)[..., None]
    yte = yte[:a.n]
    results = []
    wanted = set(a.modes.split(",")) if a.modes else None
    originals = {}
    for label, src, opts in modes(a.out, calib):
        if wanted and label not in wanted:
            continue
        gen = CodeGenerator(src, name="m_" + label.replace("-", "_"), **opts)
        g = gen.graph
        x_dev = quantize_input(g, xte)
        y, st = compile_and_run(gen, os.path.join(a.out, label), a.build, x_dev)
        row = {"mode": label, "source": gen.source_graph.source, "weights_B": gen.weight_bytes,
               "arena_B": gen.plan.arena_bytes if gen.plan else None,
               "us": st["us"], "meta_hw": st["meta_hw"], "ram_hw": st["ram_hw"],
               "features": [f.replace("PTENSOR_WITH_", "") for f in gen.features]}
        tout = g.tensors[g.outputs[0]]
        yf = y.astype(np.float32)
        if tout.dtype == "int8":
            s, z = float(tout.quant.scale[0]), int(tout.quant.zero_point[0])
            yf = (yf - z) * s
        row["accuracy"] = float((yf.argmax(1) == yte).mean())
        if g.is_quantized() and g.tensors[g.inputs[0]].dtype == "int8" and tout.dtype == "int8":
            try:
                exp = np.stack([reference.run_ptensor_int8(g, xi)[0].reshape(-1) for xi in x_dev])
                row["vs_bindings_max_lsb"] = int(np.abs(exp.astype(int) - y.astype(int)).max())
            except NotImplementedError:
                pass  # an op the Python bindings do not expose (int8 depthwise)
        elif not g.is_quantized():
            exp = reference.run_batch(g, xte)
            row["vs_numpy_max_abs"] = float(np.abs(exp - y).max())
        if src.endswith(".tflite"):
            ref = tflite_reference(src, x_dev)
            row["vs_tflite_max"] = float(np.abs(ref.astype(np.float64) - y.astype(np.float64)).max())
            row["vs_tflite_same_argmax"] = float((ref.argmax(1) == y.argmax(1)).mean())
        else:
            if src not in originals:
                originals[src] = original_output(src, xte)
            ref = originals[src]
            if ref is not None:
                row["vs_original_max_abs"] = float(np.abs(ref - yf).max())
                row["vs_original_same_argmax"] = float((ref.argmax(1) == yf.argmax(1)).mean())
        if a.cmake_check and label in CMAKE_MODES:
            st2, on, text = cmake_build_and_run(gen, os.path.join(a.out, label), x_dev)
            y2 = np.fromfile(os.path.join(a.out, label, "out_cmake.bin"), y.dtype).reshape(y.shape)
            row["cmake"] = {"features_on": on, "same_output": bool(np.array_equal(y, y2)),
                            "us": st2["us"], "text_B": text}
        results.append(row)
        print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
