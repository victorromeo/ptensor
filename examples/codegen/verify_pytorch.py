"""Train small PyTorch models on MNIST, generate pTensor C++ from them, run it,
and compare with PyTorch.

    python examples/codegen/verify_pytorch.py --out /tmp/pt_codegen

Needs torch and a CMake build of pTensor with the C API (the generated
code links build/src/pTensor/libptensor.a). The models cover the operator
families the PyTorch importer maps: convolution + batch norm, depthwise
convolution with hard-swish and squeeze-excitation, a GRU over image rows,
and a small vision transformer (patch convolution, multi-head attention,
layer norm, GELU). Each is generated as float (fused, planned arena), float
unfused with dynamic memory, and, for the CNN, int8 post-training
quantisation. Results go to --json and are summarised in PYTORCH_RESULTS.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import verify_models as vm  # noqa: E402  (compile_and_run, mnist_data)
from ptensor.codegen import CodeGenerator  # noqa: E402


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8)
        self.c2, self.b2 = nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16)
        self.fc1, self.drop, self.fc2 = nn.Linear(16 * 7 * 7, 32), nn.Dropout(0.2), nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        return self.fc2(self.drop(F.relu(self.fc1(torch.flatten(x, 1)))))


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem, self.b0 = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16)
        self.dw, self.b1 = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False), \
            nn.BatchNorm2d(16)
        self.se1, self.se2 = nn.Linear(16, 4), nn.Linear(4, 16)
        self.pw, self.b2 = nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        x = F.hardswish(self.b0(self.stem(x)))
        x = F.hardswish(self.b1(self.dw(x)))
        s = F.hardsigmoid(self.se2(F.relu(self.se1(F.adaptive_avg_pool2d(x, 1).flatten(1)))))
        x = x * s.unsqueeze(-1).unsqueeze(-1)
        x = F.relu6(self.b2(self.pw(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class RowGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(28, 48, batch_first=True)
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        out, _ = self.rnn(x.view(x.shape[0], 28, 28))
        return self.fc(out[:, -1])


class TinyViT(nn.Module):
    def __init__(self, d=24):
        super().__init__()
        self.patch = nn.Conv2d(1, d, 4, stride=4)
        self.pos = nn.Parameter(torch.zeros(1, 49, d))
        self.enc = nn.TransformerEncoderLayer(d, 3, 2 * d, dropout=0.0, activation="gelu",
                                              batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 10)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.fc(self.norm(self.enc(t)).mean(1))


MODELS = {"cnn": CNN, "mobilenet": MobileNet, "row_gru": RowGRU, "tiny_vit": TinyViT}


def train(name, xtr, ytr, out, epochs):
    path = os.path.join(out, name + ".pt")
    torch.manual_seed(0)
    m = MODELS[name]()
    if os.path.exists(path):
        m.load_state_dict(torch.load(path))
        return m.eval()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.int64))
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(m(xt[idx]), yt[idx]).backward()
            opt.step()
    torch.save(m.state_dict(), path)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ptensor_pytorch")
    ap.add_argument("--build", default=os.path.join(REPO, "build"))
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    xtr, ytr = vm.mnist_data.load("train")
    xtr = (xtr / 255.0).astype(np.float32)[:, None]
    xte, yte = vm.mnist_data.load("test")
    xte = (xte[:a.n] / 255.0).astype(np.float32)[:, None]
    yte = yte[:a.n]
    x_nhwc = xte.transpose(0, 2, 3, 1)
    rows = []
    for name in a.models.split(","):
        t0 = time.time()
        m = train(name, xtr, ytr, a.out, a.epochs)
        train_s = time.time() - t0
        with torch.no_grad():
            ref = m(torch.from_numpy(xte)).numpy()
        modes = [("float", {}), ("float-unfused-dynamic", dict(fuse=False, memory="dynamic"))]
        if name == "cnn":
            modes.append(("int8-ptq", dict(activations="int8",
                                           calibration_data=xtr[:300].transpose(0, 2, 3, 1),
                                           input_scale=1 / 255.0, input_zero_point=-128)))
        for mode, opts in modes:
            label = "%s-%s" % (name, mode)
            gen = CodeGenerator(m, name="pt_" + label.replace("-", "_"),
                                example_input=torch.from_numpy(xte[:1]), **opts)
            g = gen.graph
            x_dev = vm.quantize_input(g, x_nhwc)
            y, st = vm.compile_and_run(gen, os.path.join(a.out, label), a.build, x_dev)
            yf = y.astype(np.float32)
            tout = g.tensors[g.outputs[0]]
            if tout.dtype == "int8":
                yf = (yf - int(tout.quant.zero_point[0])) * float(tout.quant.scale[0])
            row = {"mode": label, "train_s": round(train_s, 1),
                   "nodes": len(g.nodes), "ops": g.ops_used(),
                   "weights_B": gen.weight_bytes,
                   "arena_B": gen.plan.arena_bytes if gen.plan else None,
                   "us": st["us"], "ram_hw": st["ram_hw"],
                   "accuracy_pytorch": float((ref.argmax(1) == yte).mean()),
                   "accuracy_ptensor": float((yf.argmax(1) == yte).mean()),
                   "max_abs_vs_pytorch": float(np.abs(yf - ref).max()),
                   "same_argmax": float((yf.argmax(1) == ref.argmax(1)).mean())}
            print(json.dumps(row), flush=True)
            rows.append(row)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
