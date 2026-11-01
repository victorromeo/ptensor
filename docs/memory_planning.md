# Arena memory planning

`ArenaPlanner` (`src/pTensor/allocators/arenaPlanner.hpp`) works out, before
inference starts, where every activation tensor lives in one block of RAM.
Tensors that are never live at the same time share bytes, and elementwise
operators can write their output over their input. The block's size is the
whole activation RAM the model needs; nothing is allocated, moved or freed
while the model runs.

## Use

```cpp
#include "pTensor.h"
using namespace pTensor;

ArenaPlanner<16> plan;                         // up to 16 tensors, 8-byte aligned
uint16_t x  = plan.add_tensor(28 * 28);        // sizes in bytes
uint16_t c1 = plan.add_tensor(28 * 28 * 8);
uint16_t r1 = plan.add_tensor(28 * 28 * 8);
uint16_t p1 = plan.add_tensor(14 * 14 * 8);
plan.add_op(x, c1);                            // conv: reads x, writes c1
plan.add_inplace_op(c1, r1);                   // relu: r1 may reuse c1's bytes
plan.add_op(r1, p1);                           // maxpool
plan.mark_output(p1);                          // read after the last operator
plan.plan();

// One block, from a static array or from the RAM arena allocator.
uint8_t* arena = (uint8_t*)ram_arena.allocate(plan.arena_bytes());
Tensor tx = new BufferTensor({1, 28, 28, 1}, i8, plan.place(x, arena));
```

Operators are added in execution order. Operators with several inputs or
outputs use `add_op(inputs, n, outputs, m)`; `add_inplace_op(inputs, n, out)`
lets `out` reuse `inputs[0]`.

What it computes:

| Call | Meaning |
| --- | --- |
| `arena_bytes()` | Size of the block the plan needs. |
| `offset(id)`, `place(id, arena)` | Where tensor `id` lives. |
| `naive_bytes()` | RAM if every tensor had its own buffer. |
| `live_peak_bytes()` | Most bytes live at any one time without in-place sharing; no placement can beat it without in-place operators. |
| `inplace_count()`, `shares_buffer(id)` | Which in-place requests were accepted. |

The planner uses no heap. `ArenaPlanner<N, Alignment>` keeps its tables
inside the object (22 bytes per tensor), and the code is about 1.5 KB of
Cortex-M4 flash. It is only linked into firmware that uses it. You can run
the plan once at boot, or run it on the host and put the offsets in the
firmware as constants.

## How it decides

**Lifetimes.** A tensor is live from the operator that first writes it to the
last operator that reads it. A tensor that is read before anything writes it
is a graph input, so it is live from the start. A tensor passed to
`mark_output` stays live past the last operator.

**In-place.** An elementwise operator reads element *i* before it writes
element *i*, so its output can sit on top of its input. `plan()` accepts an
`add_inplace_op` request only when all of these hold:

- this operator is the last reader of everything already in that buffer,
- the input is not a graph output, and
- the output is no bigger than the input.

A narrowing operator, such as float to int8 quantize, writes byte *i* only
after it has read bytes *4i* to *4i+3*, so it qualifies. Chains of these
operators fold into one buffer, for example conv → ReLU → quantize.

Operators that are safe to declare in-place: ReLU and ReLU6 (float and int8),
every unary activation in `ElementwiseOperators.hpp`, float softmax (it reads
a whole row before rewriting it), the broadcast binary operators when the
aliased input has the output's shape, `BatchNorm`, `ChannelAffine`,
`Quantize`, and reshape or squeeze, which only reinterpret the buffer. `tests/allocators/test_arenaPlanner.cpp` runs a representative
set of these with the output written over the input and checks that the
results are bit-identical to running them out of place. Convolution, pooling,
dense, transpose, pad, concat and the int8 softmax read elements after they
write others, so they are not in-place safe.

**Placement.** Greedy by size, the heuristic TFLite Micro uses. The largest
buffer goes first, and each buffer is placed at the lowest offset that does
not collide with an already placed buffer whose lifetime overlaps its own. On
chain-shaped networks it reaches the lower bound.

## MNIST, before and after

The `examples/memory_plan` runner executes the int8 MNIST model
(`examples/mnist/mnist_int8.tflite`). The network is conv 3x3x8 → maxpool →
conv 3x3x16 → maxpool → dense 32 → dense 10. Each image runs twice, once with
pTensor's dynamic RamTensors and once in the planned arena. The logits are
identical for all 10,000 test images, and accuracy is 97.81%.

| Activation memory | Bytes |
| --- | --- |
| One buffer per activation | 12,588 |
| Ping-pong, 2 x largest activation (what `examples/mnist_modes` does) | 12,544 |
| Dynamic RamTensors in a 64 KB `localCircularArenaAllocator` (high-water mark) | 29,456 |
| Dynamic RamTensors in an 8 KB arena (fits, but compacts live tensors 3 times per inference) | 7,840 |
| **Planned arena** | **7,840** (−37.5% vs ping-pong) |

The same network in TFLite's unfused form has ReLU, RESHAPE and SOFTMAX as
operators of their own. Planning that form gives:

| | Bytes |
| --- | --- |
| One buffer per activation | 22,824 |
| Planned, no in-place | 12,544 |
| Planned, in-place where safe (4 operators) | 7,840 |

The dynamic arena can only reach the planned size by trial and error, and it
pays for it with run-time copying. The planner gets the minimum up front with
no copying, and gives a size you can check at build time.

## Related fix

Assigning a new tensor to a `Tensor` handle (`t = new RamTensor(...)` or
`t = std::move(other)`) used to leak the tensor the handle held. A model that
kept one handle per layer used up its RAM arena after a few inferences. The
handle now frees the old tensor first.
