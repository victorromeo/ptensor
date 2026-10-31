"""Download and cache the MNIST test set (IDX files), using only NumPy."""
import gzip
import os
import urllib.request

import numpy as np

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def cache_dir():
    root = os.environ.get("PTENSOR_MNIST_DIR")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "ptensor", "mnist")
    os.makedirs(root, exist_ok=True)
    return root


def _fetch(name):
    path = os.path.join(cache_dir(), name)
    if os.path.exists(path):
        return path
    errors = []
    for base in MIRRORS:
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(base + name, tmp)
            os.replace(tmp, path)
            return path
        except Exception as exc:  # try the next mirror
            errors.append("%s: %s" % (base, exc))
    raise RuntimeError("could not download %s\n  %s" % (name, "\n  ".join(errors)))


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(split="test"):
    """Return (images uint8 [N,28,28], labels uint8 [N]) for 'train' or 'test'."""
    images = _read_idx(_fetch(FILES[split + "_images"]))
    labels = _read_idx(_fetch(FILES[split + "_labels"]))
    return images, labels
