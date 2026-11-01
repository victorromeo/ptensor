#include "arenaPlanner.hpp"

namespace pTensor {

const uint16_t ArenaPlannerBase::kInvalid;

ArenaPlannerBase::ArenaPlannerBase(Entry* entries, uint16_t* order,
                                   uint16_t max_tensors, uint32_t alignment)
    : _entries(entries),
      _order(order),
      _max(max_tensors),
      _num(0),
      _num_ops(0),
      _align(alignment ? alignment : 1),
      _arena_bytes(0),
      _live_peak(0),
      _inplace_count(0) {}

void ArenaPlannerBase::reset() {
  _num = 0;
  _num_ops = 0;
  _arena_bytes = 0;
  _live_peak = 0;
  _inplace_count = 0;
}

uint16_t ArenaPlannerBase::add_tensor(uint32_t bytes) {
  if (_num >= _max) return kInvalid;
  Entry& e = _entries[_num];
  e.size = bytes;
  e.offset = 0;
  e.first = -1;
  e.last = -1;
  e.alias = kInvalid;
  e.alias_op = -1;
  e.group_last = -1;
  e.flags = 0;
  return _num++;
}

void ArenaPlannerBase::touch_read(uint16_t id, int op) {
  Entry& e = _entries[id];
  // A tensor read before anything wrote it is a graph input: it is live from
  // before the first operator (first stays -1).
  e.flags |= kRead;
  if (op > e.last) e.last = static_cast<int16_t>(op);
}

void ArenaPlannerBase::touch_write(uint16_t id, int op) {
  Entry& e = _entries[id];
  if (!(e.flags & (kRead | kWritten))) e.first = static_cast<int16_t>(op);
  e.flags |= kWritten;
  // An output nobody reads still occupies its bytes while the op runs.
  if (op > e.last) e.last = static_cast<int16_t>(op);
}

int ArenaPlannerBase::add_op(const uint16_t* inputs, size_t n_inputs,
                             const uint16_t* outputs, size_t n_outputs) {
  for (size_t i = 0; i < n_inputs; ++i)
    if (inputs[i] >= _num) return -1;
  for (size_t i = 0; i < n_outputs; ++i)
    if (outputs[i] >= _num) return -1;
  if (_num_ops >= 0x7FFF) return -1;
  const int op = _num_ops++;
  for (size_t i = 0; i < n_inputs; ++i) touch_read(inputs[i], op);
  for (size_t i = 0; i < n_outputs; ++i) touch_write(outputs[i], op);
  return op;
}

int ArenaPlannerBase::add_inplace_op(const uint16_t* inputs, size_t n_inputs,
                                     uint16_t output) {
  const int op = add_op(inputs, n_inputs, &output, 1);
  if (op >= 0 && n_inputs > 0 && inputs[0] != output) {
    _entries[output].alias = inputs[0];
    _entries[output].alias_op = static_cast<int16_t>(op);
  }
  return op;
}

void ArenaPlannerBase::mark_output(uint16_t id) {
  if (id < _num) _entries[id].flags |= kOutput;
}

// Whether two placed buffers are live at the same time.
static inline bool overlaps(const ArenaPlannerBase::Entry& a,
                            const ArenaPlannerBase::Entry& b) {
  return a.first <= b.group_last && b.first <= a.group_last;
}

uint16_t ArenaPlannerBase::root(uint16_t id) const {
  while (_entries[id].flags & kShared) id = _entries[id].alias;
  return id;
}

uint32_t ArenaPlannerBase::offset(uint16_t id) const {
  return id < _num ? _entries[id].offset : 0;
}

bool ArenaPlannerBase::shares_buffer(uint16_t id) const {
  return id < _num && (_entries[id].flags & kShared);
}

uint32_t ArenaPlannerBase::naive_bytes() const {
  uint32_t total = 0;
  for (uint16_t i = 0; i < _num; ++i) total += aligned(_entries[i].size);
  return total;
}

bool ArenaPlannerBase::plan() {
  bool all_used = true;
  const int16_t end = static_cast<int16_t>(_num_ops);
  for (uint16_t i = 0; i < _num; ++i) {
    Entry& e = _entries[i];
    e.flags &= static_cast<uint8_t>(~(kShared | kPlaced));
    if (!(e.flags & (kRead | kWritten))) all_used = false;
    if (e.flags & kOutput) e.last = end;
    e.group_last = e.last;
    e.offset = 0;
  }

  // Peak of simultaneously live bytes, before any sharing.
  _live_peak = 0;
  for (int t = -1; t <= _num_ops; ++t) {
    uint32_t live = 0;
    for (uint16_t i = 0; i < _num; ++i)
      if (_entries[i].first <= t && t <= _entries[i].last)
        live += aligned(_entries[i].size);
    if (live > _live_peak) _live_peak = live;
  }

  // In-place sharing, in execution order so that chains (conv -> relu ->
  // quantize ...) fold into one buffer.
  _inplace_count = 0;
  uint16_t n_alias = 0;
  for (uint16_t i = 0; i < _num; ++i)
    if (_entries[i].alias != kInvalid) _order[n_alias++] = i;
  for (uint16_t i = 1; i < n_alias; ++i) {
    const uint16_t v = _order[i];
    uint16_t j = i;
    while (j > 0 && _entries[_order[j - 1]].alias_op > _entries[v].alias_op) {
      _order[j] = _order[j - 1];
      --j;
    }
    _order[j] = v;
  }
  for (uint16_t k = 0; k < n_alias; ++k) {
    Entry& out = _entries[_order[k]];
    const Entry& in = _entries[out.alias];
    Entry& r = _entries[root(out.alias)];
    // Safe only when this operator is the last reader of everything already
    // living in the buffer, and the output fits in the input's bytes (an
    // elementwise op writes element i no earlier than it reads element i).
    if (r.group_last != out.alias_op || (in.flags & kOutput) ||
        out.size > in.size || out.first != out.alias_op)
      continue;
    out.flags |= kShared;
    if (out.group_last > r.group_last) r.group_last = out.group_last;
    ++_inplace_count;
  }

  // Greedy by size: biggest buffers first, each at the lowest offset that is
  // free for its whole lifetime.
  uint16_t n_place = 0;
  for (uint16_t i = 0; i < _num; ++i)
    if (!(_entries[i].flags & kShared)) _order[n_place++] = i;
  for (uint16_t i = 1; i < n_place; ++i) {
    const uint16_t v = _order[i];
    const Entry& ev = _entries[v];
    uint16_t j = i;
    while (j > 0) {
      const Entry& ep = _entries[_order[j - 1]];
      const bool before = ep.size > ev.size ||
                          (ep.size == ev.size && ep.first <= ev.first);
      if (before) break;
      _order[j] = _order[j - 1];
      --j;
    }
    _order[j] = v;
  }

  _arena_bytes = 0;
  for (uint16_t k = 0; k < n_place; ++k) {
    Entry& e = _entries[_order[k]];
    const uint32_t sz = aligned(e.size);
    uint32_t best = 0;
    if (sz > 0) {
      // Candidate offsets are 0 and the end of each buffer that is live at the
      // same time; the lowest candidate that collides with none of them wins.
      best = 0xFFFFFFFFu;
      for (uint16_t c = 0; c <= k; ++c) {
        uint32_t cand = 0;
        if (c < k) {
          const Entry& p = _entries[_order[c]];
          if (!overlaps(p, e)) continue;
          cand = p.offset + aligned(p.size);
        }
        if (cand >= best) continue;
        bool ok = true;
        for (uint16_t q = 0; q < k && ok; ++q) {
          const Entry& p = _entries[_order[q]];
          if (!overlaps(p, e)) continue;
          const uint32_t psz = aligned(p.size);
          if (psz && cand < p.offset + psz && p.offset < cand + sz) ok = false;
        }
        if (ok) best = cand;
      }
    }
    e.offset = best;
    e.flags |= kPlaced;
    if (best + sz > _arena_bytes) _arena_bytes = best + sz;
  }
  for (uint16_t i = 0; i < _num; ++i)
    if (_entries[i].flags & kShared)
      _entries[i].offset = _entries[root(i)].offset;
  return all_used;
}

}  // namespace pTensor
