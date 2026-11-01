#ifndef PTENSOR_ARENA_PLANNER_HPP
#define PTENSOR_ARENA_PLANNER_HPP
/**
 * Static memory planning for activation tensors.
 *
 * A network's intermediate tensors do not all live at once: the output of
 * layer 3 is dead once layer 4 has read it. The planner takes the list of
 * tensors (their sizes) and operators (which tensors each one reads and
 * writes), works out when each tensor is first written and last read, and
 * gives every tensor an offset in one arena so that tensors whose lifetimes
 * overlap never share bytes. The arena is then a single block whose size is
 * known before inference starts: no allocation, rebalancing or
 * fragmentation happens while the network runs.
 *
 *   ArenaPlanner<16> plan;
 *   uint16_t x  = plan.add_tensor(784);       // graph input
 *   uint16_t c1 = plan.add_tensor(5408);
 *   uint16_t r1 = plan.add_tensor(5408);
 *   plan.add_op(x, c1);                        // conv
 *   plan.add_inplace_op(c1, r1);               // relu, may reuse c1's bytes
 *   ...
 *   plan.mark_output(logits);
 *   plan.plan();
 *   static uint8_t arena[...];                 // or allocate plan.arena_bytes()
 *   Tensor t = new BufferTensor(shape, i8, plan.place(c1, arena));
 *
 * Placement uses the greedy-by-size heuristic (largest tensor first, each at
 * the lowest offset that does not collide with an already placed tensor whose
 * lifetime overlaps), which is what TFLite Micro uses and is close to optimal
 * on the chain-shaped graphs typical of microcontroller models.
 *
 * In-place operators: an elementwise operator (ReLU, sigmoid, add, batch
 * norm, reshape, ...) reads element i before it writes element i, so its
 * output may occupy the same bytes as its input. add_inplace_op() records
 * that the output *may* share the input's buffer; plan() makes it share only
 * when that is safe, which is when this operator is the input's last reader,
 * the input is not a graph output, and the output is no larger than the
 * input.
 *
 * The planner never touches the heap: the tensor table lives in the
 * ArenaPlanner<N> object, so it can be a static on a microcontroller, or the
 * plan can be computed on the host and the offsets baked into the firmware.
 */
#include <stddef.h>
#include <stdint.h>

namespace pTensor {

class ArenaPlannerBase {
 public:
  static const uint16_t kInvalid = 0xFFFF;

  struct Entry {
    uint32_t size;      // requested bytes
    uint32_t offset;    // planned offset (valid after plan())
    int16_t first;      // first operator that writes it (-1: graph input)
    int16_t last;       // last operator that reads it
    uint16_t alias;     // tensor whose buffer this one may share, or kInvalid
    int16_t alias_op;   // operator that would make the alias
    int16_t group_last; // last use of everything sharing this buffer
    uint8_t flags;
  };

  /** Declare a tensor of `bytes` bytes. Returns its id, or kInvalid when the
   * table is full. */
  uint16_t add_tensor(uint32_t bytes);

  /** Record an operator that reads `inputs` and writes `outputs`. Operators
   * must be added in execution order. Returns the operator index, or -1 if an
   * id is invalid. */
  int add_op(const uint16_t* inputs, size_t n_inputs, const uint16_t* outputs,
             size_t n_outputs);
  /** Convenience form for the common one-input, one-output operator. */
  int add_op(uint16_t input, uint16_t output) {
    return add_op(&input, 1, &output, 1);
  }

  /** Like add_op, but `output` may be written over `inputs[0]` (an
   * elementwise operator whose output has the same layout as that input).
   * Other inputs are read normally. */
  int add_inplace_op(const uint16_t* inputs, size_t n_inputs, uint16_t output);
  int add_inplace_op(uint16_t input, uint16_t output) {
    return add_inplace_op(&input, 1, output);
  }

  /** The tensor is read after the last operator (it stays live to the end). */
  void mark_output(uint16_t id);

  /** Compute the lifetimes, resolve in-place sharing and place every tensor.
   * Returns false if a tensor is never written or read (it is still placed). */
  bool plan();

  /** Bytes of arena the plan needs (valid after plan()). */
  uint32_t arena_bytes() const { return _arena_bytes; }
  /** Bytes needed if every tensor had its own buffer. */
  uint32_t naive_bytes() const;
  /** Most bytes that are live at the same time, ignoring fragmentation and
   * in-place sharing: no placement can do better than this without in-place
   * operators. */
  uint32_t live_peak_bytes() const { return _live_peak; }
  /** Number of in-place shares plan() accepted. */
  uint16_t inplace_count() const { return _inplace_count; }

  uint32_t offset(uint16_t id) const;
  /** Whether plan() made `id` share another tensor's buffer. */
  bool shares_buffer(uint16_t id) const;
  /** Pointer to tensor `id` inside an arena of at least arena_bytes(). */
  void* place(uint16_t id, void* arena) const {
    return static_cast<uint8_t*>(arena) + offset(id);
  }

  uint16_t num_tensors() const { return _num; }
  int num_ops() const { return _num_ops; }
  const Entry& entry(uint16_t id) const { return _entries[id]; }
  uint32_t alignment() const { return _align; }

  /** Forget every tensor and operator. */
  void reset();

 protected:
  ArenaPlannerBase(Entry* entries, uint16_t* order, uint16_t max_tensors,
                   uint32_t alignment);

 private:
  enum Flags : uint8_t { kOutput = 1, kShared = 2, kPlaced = 4, kRead = 8, kWritten = 16 };
  uint16_t root(uint16_t id) const;
  void touch_read(uint16_t id, int op);
  void touch_write(uint16_t id, int op);
  uint32_t aligned(uint32_t bytes) const {
    return (bytes + _align - 1) / _align * _align;
  }

  Entry* _entries;
  uint16_t* _order;
  uint16_t _max;
  uint16_t _num;
  int _num_ops;
  uint32_t _align;
  uint32_t _arena_bytes;
  uint32_t _live_peak;
  uint16_t _inplace_count;
};

/** A planner for at most MaxTensors tensors. Each tensor costs 22 bytes of RAM. */
template <uint16_t MaxTensors, uint32_t Alignment = 8>
class ArenaPlanner : public ArenaPlannerBase {
  static_assert(MaxTensors > 0 && MaxTensors < kInvalid,
                "ArenaPlanner needs between 1 and 65534 tensors");
  static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                "ArenaPlanner alignment must be a power of two");

 public:
  ArenaPlanner()
      : ArenaPlannerBase(_storage, _order_storage, MaxTensors, Alignment) {}

 private:
  Entry _storage[MaxTensors];
  uint16_t _order_storage[MaxTensors];
};

}  // namespace pTensor
#endif
