"""python -m ptensor.codegen MODEL -o OUT_DIR [options]"""
import argparse
import sys

import numpy as np

from .generator import CodeGenerator


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ptensor.codegen",
                                description="Generate pTensor C/C++ from a model.")
    p.add_argument("model", help=".tflite, .keras / .h5, or a SavedModel directory")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--name", help="C identifier prefix (default: from the model)")
    p.add_argument("--framework", help="force a frontend (keras, tflite, tf, pytorch, onnx)")
    p.add_argument("--weights", default="float32",
                   help="weight storage with float activations: float32, float16, int8, "
                        "uint2..uint7, hough2/3/4, quadN, cubicN")
    p.add_argument("--activations", default="float32", choices=("float32", "int8"))
    p.add_argument("--calibration", help=".npy of float inputs for --activations int8")
    p.add_argument("--input-scale", type=float)
    p.add_argument("--input-zero-point", type=int)
    p.add_argument("--float-io", action="store_true", help="int8 model with float in/out")
    p.add_argument("--no-fuse", action="store_true", help="run every node unfused")
    p.add_argument("--no-fast-kernels", action="store_true")
    p.add_argument("--weights-in-ram", action="store_true",
                   help="decode compressed weights to RAM at init")
    p.add_argument("--prune", type=float, default=0.0, help="fraction of channels to remove")
    p.add_argument("--memory", default="planned", choices=("planned", "dynamic"))
    p.add_argument("--alignment", type=int, default=8)
    a = p.parse_args(argv)
    calib = np.load(a.calibration) if a.calibration else None
    gen = CodeGenerator(a.model, framework=a.framework, name=a.name, weights=a.weights,
                        activations=a.activations, calibration_data=calib,
                        input_scale=a.input_scale, input_zero_point=a.input_zero_point,
                        float_io=a.float_io, fuse=not a.no_fuse,
                        fast_kernels=not a.no_fast_kernels, weights_in_ram=a.weights_in_ram,
                        prune=a.prune, memory=a.memory, alignment=a.alignment)
    for path in gen.generate(a.out):
        print("wrote", path)
    print(gen.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
