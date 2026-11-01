# Arena memory planning report

Runs an int8 `.ptm` model twice per image, once with dynamically allocated
RamTensors and once from an `ArenaPlanner` arena. It checks that both runs
give the same logits and prints the activation RAM each approach needs. See
[docs/memory_planning.md](../../docs/memory_planning.md) for the planner
itself and the results.

```sh
# from the repository root
pip install numpy tflite
python3 examples/memory_plan/export_mnist.py /tmp/mp     # MNIST int8 model + test set
cmake -S . -B build -DPTENSOR_MEMORY_PLAN=ON && cmake --build build --target memory_plan
./build/examples/memory_plan/memory_plan /tmp/mp/mnist_int8.ptm /tmp/mp/mnist_test.bin
```

`--limit N` runs only the first N images. Build with
`-DCMAKE_CXX_FLAGS=-DDYNAMIC_ARENA_BYTES=8192` to squeeze the dynamic arena to
roughly the planned size and count how often it has to compact.

Output on the full test set:

```
Activations (int8 bytes, planned offset):
  input              784  @6272
  conv    28x28x8     6272  @0
  maxpool 14x14x8     1568  @6272
  conv    14x14x16    3136  @0
  maxpool  7x 7x16     784  @3136
  dense    1x 1x32      32  @0
  dense    1x 1x10      10  @32

Peak activation RAM, this network (6 layers):
  one buffer per activation (naive)         12588 bytes
  ping-pong, 2 x largest activation         12544 bytes
  dynamic RamTensors, arena high-water      29456 bytes  (65536-byte arena, 0.0 compactions per inference)
  planned arena                              7840 bytes
  lower bound, most bytes live at once       7840 bytes

Same network with unfused ReLU/RESHAPE/SOFTMAX operators (planning only):
  one buffer per activation (naive)         22824 bytes
  planned, no in-place                      12544 bytes
  planned, in-place where safe ( 4 ops)      7840 bytes

Images 10000, accuracy 97.81%, planned vs dynamic logits differ on 0
```
