// Peak activation RAM of an int8 .ptm model, before and after arena planning.
//
//   memory_plan model.ptm test.bin [--limit N]
//
// The model runs twice per image:
//   dynamic  every activation is a RamTensor allocated from a
//            localCircularArenaAllocator when its layer runs and freed as soon
//            as the next layer has read it (pTensor's default way). The
//            arena's high-water mark is the RAM this needs.
//   planned  ArenaPlanner works out every activation's lifetime once, places
//            them all in one block taken from the RAM arena, and every
//            activation is a BufferTensor at its planned offset.
// Both runs must give bit-identical logits. The report also gives the RAM a
// naive layout (one buffer per activation) and a ping-pong layout (two
// buffers of the largest activation) would need, and plans the same network
// in TFLite's unfused form (separate ReLU, RESHAPE and SOFTMAX operators) to
// show what in-place sharing saves there.
//
// examples/memory_plan/export_mnist.py writes the MNIST model and test set.
// test.bin: u32 count, u16 h, u16 w, u16 c, count*h*w*c u8 pixels, count u8 labels.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

localCircularArenaAllocator<16 * 1024, uint32_t, 128> g_meta;
// RAM arena for the dynamic run's RamTensors. Build with
// -DDYNAMIC_ARENA_BYTES=8192 to see it squeezed to (nearly) the planned size,
// where it has to compact live tensors during every inference to fit.
#ifndef DYNAMIC_ARENA_BYTES
#define DYNAMIC_ARENA_BYTES (64 * 1024)
#endif
template <size_t N>
struct CountingArena : public localCircularArenaAllocator<N, uint32_t, 64> {
  uint32_t rebalances = 0;
  virtual bool rebalance() override {
    ++rebalances;
    return localCircularArenaAllocator<N, uint32_t, 64>::rebalance();
  }
};
CountingArena<DYNAMIC_ARENA_BYTES> g_ram;
// The arena the planned block is taken from. It is separate only so the two
// runs can be measured independently; a firmware would use one.
localCircularArenaAllocator<64 * 1024, uint32_t, 4> g_plan_ram;
SimpleErrorHandler g_err(4);
bool g_failed = false;

// Plans use 4-byte alignment, what a Cortex-M needs for int32/float data.
typedef ArenaPlanner<32, 4> Planner;

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  template <typename T>
  T get() {
    T v;
    if (p + sizeof(T) > end) throw "truncated file";
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  const uint8_t* bytes(size_t n) {
    if (p + n > end) throw "truncated file";
    const uint8_t* r = p;
    p += n;
    return r;
  }
};

enum { ST_I8 = 3, ST_I32 = 4 };
enum Op { OP_CONV = 1, OP_POOL = 2, OP_DENSE = 3 };

struct TensorRec {
  uint8_t storage = 0;
  std::vector<uint16_t> dims;
  std::vector<uint32_t> aligned;
  TensorShape shape() const {
    switch (dims.size()) {
      case 1: return TensorShape(dims[0]);
      case 2: return TensorShape(dims[0], dims[1]);
      case 3: return TensorShape(dims[0], dims[1], dims[2]);
      default: return TensorShape(dims[0], dims[1], dims[2], dims[3]);
    }
  }
  Tensor make() const {
    return new RomTensor(shape(), storage == ST_I8 ? i8 : i32, aligned.data());
  }
};

TensorRec read_tensor(Reader& r) {
  TensorRec t;
  t.storage = r.get<uint8_t>();
  if (t.storage != ST_I8 && t.storage != ST_I32) throw "only int8/int32 weights";
  const uint8_t rank = r.get<uint8_t>();
  for (uint8_t i = 0; i < rank; ++i) t.dims.push_back(r.get<uint16_t>());
  const uint32_t n = r.get<uint32_t>();
  t.aligned.resize((n + 3) / 4);
  memcpy(t.aligned.data(), r.bytes(n), n);
  return t;
}

struct Layer {
  uint8_t op = 0, act = 0, padding = 2, kh = 0, kw = 0, sh = 1, sw = 1;
  float out_scale = 1.0f;
  int32_t out_zp = 0;
  std::vector<float> w_scales;
  std::vector<int32_t> w_zps;
  TensorRec w, b;
  uint16_t oh = 1, ow = 1, oc = 1;  // output shape
  uint32_t out_count() const { return uint32_t(oh) * ow * oc; }
};

struct Model {
  std::vector<uint8_t> file;
  uint16_t in_h = 0, in_w = 0, in_c = 0;
  float in_scale = 1.0f;
  int32_t in_zp = 0;
  std::vector<Layer> layers;
};

bool load_file(const char* path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  out.resize(ftell(f));
  fseek(f, 0, SEEK_SET);
  const bool ok = fread(out.data(), 1, out.size(), f) == out.size();
  fclose(f);
  return ok;
}

uint16_t out_size(uint16_t in, uint16_t k, uint16_t s, uint8_t padding) {
  if (padding == SAME) return (in + s - 1) / s;
  return (in - k) / s + 1;
}

void parse(Model& m) {
  Reader r = {m.file.data(), m.file.data() + m.file.size()};
  if (memcmp(r.bytes(4), "PTM1", 4) != 0) throw "not a .ptm file";
  if (r.get<uint8_t>() != 1) throw "only int8 (.ptm kind 1) models";
  m.in_h = r.get<uint16_t>();
  m.in_w = r.get<uint16_t>();
  m.in_c = r.get<uint16_t>();
  m.in_scale = r.get<float>();
  m.in_zp = r.get<int32_t>();
  uint16_t h = m.in_h, w = m.in_w, c = m.in_c;
  const uint32_t n = r.get<uint32_t>();
  for (uint32_t i = 0; i < n; ++i) {
    Layer l;
    l.op = r.get<uint8_t>();
    if (l.op == OP_CONV) {
      l.act = r.get<uint8_t>();
      l.padding = r.get<uint8_t>();
      l.sh = r.get<uint8_t>();
      l.sw = r.get<uint8_t>();
    } else if (l.op == OP_POOL) {
      l.kh = r.get<uint8_t>();
      l.kw = r.get<uint8_t>();
      l.sh = r.get<uint8_t>();
      l.sw = r.get<uint8_t>();
      l.padding = r.get<uint8_t>();
    } else if (l.op == OP_DENSE) {
      l.act = r.get<uint8_t>();
    } else {
      throw "unknown layer";
    }
    if (l.op != OP_POOL) {
      l.out_scale = r.get<float>();
      l.out_zp = r.get<int32_t>();
      const uint32_t ns = r.get<uint32_t>();
      for (uint32_t j = 0; j < ns; ++j) l.w_scales.push_back(r.get<float>());
      l.w_zps.assign(ns, 0);
      l.w = read_tensor(r);
      l.b = read_tensor(r);
    }
    if (l.op == OP_CONV) {
      l.oh = out_size(h, l.w.dims[1], l.sh, l.padding);
      l.ow = out_size(w, l.w.dims[2], l.sw, l.padding);
      l.oc = l.w.dims[0];
    } else if (l.op == OP_POOL) {
      l.oh = out_size(h, l.kh, l.sh, l.padding);
      l.ow = out_size(w, l.kw, l.sw, l.padding);
      l.oc = c;
    } else {
      l.oh = 1;
      l.ow = 1;
      l.oc = l.w.dims[1];
    }
    h = l.oh;
    w = l.ow;
    c = l.oc;
    m.layers.push_back(l);
  }
}

// ---- planning ---------------------------------------------------------------

// The network as the runner executes it: one tensor per layer output, plus the
// input. ids[0] is the input, ids[i + 1] layer i's output.
void plan_runner_graph(const Model& m, Planner& p, std::vector<uint16_t>& ids) {
  ids.clear();
  ids.push_back(p.add_tensor(uint32_t(m.in_h) * m.in_w * m.in_c));
  for (const Layer& l : m.layers) {
    const uint16_t out = p.add_tensor(l.out_count());
    p.add_op(ids.back(), out);
    ids.push_back(out);
  }
  p.mark_output(ids.back());
  p.plan();
}

// The same network as TFLite's converter emits it before fusion: activation
// functions, the flatten and the softmax are operators of their own. With
// `inplace` the activations and the flatten are declared in-place capable.
void plan_unfused_graph(const Model& m, Planner& p, bool inplace) {
  uint16_t cur = p.add_tensor(uint32_t(m.in_h) * m.in_w * m.in_c);
  bool flat = false;
  for (const Layer& l : m.layers) {
    if (l.op == OP_DENSE && !flat) {
      const uint16_t r = p.add_tensor(p.entry(cur).size);  // RESHAPE
      inplace ? p.add_inplace_op(cur, r) : p.add_op(cur, r);
      cur = r;
      flat = true;
    }
    uint16_t out = p.add_tensor(l.out_count());
    p.add_op(cur, out);
    cur = out;
    if (l.act) {
      out = p.add_tensor(l.out_count());  // RELU / RELU6
      inplace ? p.add_inplace_op(cur, out) : p.add_op(cur, out);
      cur = out;
    }
  }
  // SOFTMAX. The int8 kernel re-reads its input after writing outputs, so it
  // is not in-place safe (the float one is).
  const uint16_t sm = p.add_tensor(p.entry(cur).size);
  p.add_op(cur, sm);
  p.mark_output(sm);
  p.plan();
}

// ---- inference --------------------------------------------------------------

struct Placement {
  const Planner* plan;      // null: allocate RamTensors from the RAM arena
  const uint16_t* ids;
  uint8_t* arena;
  TensorInterface* make(size_t idx, const TensorShape& s) const {
    if (!plan) return new RamTensor(s, i8);
    return new BufferTensor(s, i8, plan->place(ids[idx], arena));
  }
};

void run(const Model& m, const int8_t* input, const Placement& where,
         int8_t* logits) {
  uint16_t h = m.in_h, w = m.in_w, c = m.in_c;
  float in_scale = m.in_scale;
  int32_t in_zp = m.in_zp;
  Tensor cur = where.make(0, {1, h, w, c});
  memcpy(cur->raw_mutable_data(), input, size_t(h) * w * c);
  for (size_t li = 0; li < m.layers.size(); ++li) {
    const Layer& l = m.layers[li];
    Tensor next = where.make(li + 1, l.op == OP_DENSE
                                         ? TensorShape(1, l.oc)
                                         : TensorShape(1, l.oh, l.ow, l.oc));
    {
      // A view of the current activation in the shape this layer reads.
      Tensor x = l.op == OP_DENSE
                     ? new BufferTensor({1, uint16_t(h * w * c)}, i8,
                                        cur->raw_mutable_data())
                     : new BufferTensor({1, h, w, c}, i8, cur->raw_mutable_data());
      if (l.op == OP_CONV) {
        typedef TflmSymQuantOps::Conv2dOperator<int8_t> Conv;
        x->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
        Tensor f = l.w.make();
        f->set_quantization_params(PerChannelQuantizationParams(
            l.w_zps.data(), l.w_scales.data(), (int)l.w_scales.size()));
        Tensor bias = l.b.make();
        next->set_quantization_params(PerTensorQuantizationParams(l.out_zp, l.out_scale));
        const uint16_t strides[2] = {l.sh, l.sw};
        const uint16_t dilation[2] = {1, 1};
        Conv op(strides, static_cast<Padding>(l.padding), dilation,
                static_cast<TFLM::TfLiteFusedActivation>(l.act));
        op.set_inputs({{Conv::in, x}, {Conv::filter, f}, {Conv::bias, bias}})
            .set_outputs({{Conv::out, next}}).eval();
      } else if (l.op == OP_POOL) {
        typedef MaxPoolOperator<int8_t> Pool;
        Pool op({l.kh, l.kw}, {1, l.sh, l.sw, 1}, static_cast<Padding>(l.padding));
        op.set_inputs({{Pool::in, x}}).set_outputs({{Pool::out, next}}).eval();
      } else {
        typedef TflmSymQuantOps::FullyConnectedOperator<int8_t> FC;
        x->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
        Tensor wt = l.w.make();
        wt->set_quantization_params(PerChannelQuantizationParams(
            l.w_zps.data(), l.w_scales.data(), (int)l.w_scales.size()));
        Tensor bias = l.b.make();
        next->set_quantization_params(PerTensorQuantizationParams(l.out_zp, l.out_scale));
        FC op(static_cast<TFLM::TfLiteFusedActivation>(l.act));
        op.set_inputs({{FC::input, x}, {FC::filter, wt}, {FC::bias, bias}})
            .set_outputs({{FC::output, next}}).eval();
      }
    }
    if (l.op != OP_POOL) {
      in_scale = l.out_scale;
      in_zp = l.out_zp;
    }
    h = l.oh;
    w = l.ow;
    c = l.oc;
    cur = std::move(next);  // the layer's input is dead: free it
  }
  memcpy(logits, cur->raw_data(), m.layers.back().out_count());
}

uint32_t argmax(const int8_t* v, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.ptm test.bin [--limit N]\n", argv[0]);
    return 2;
  }
  uint32_t limit = 0xFFFFFFFFu;
  for (int i = 3; i < argc; ++i)
    if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);

  Context* ctx = Context::get_default_context();
  ctx->set_metadata_allocator(&g_meta);
  ctx->set_ram_data_allocator(&g_ram);
  g_err.set_onError([](Error*) { g_failed = true; });
  ctx->set_ErrorHandler(&g_err);

  Model m;
  std::vector<uint8_t> test;
  if (!load_file(argv[1], m.file) || !load_file(argv[2], test)) {
    fprintf(stderr, "cannot read inputs\n");
    return 1;
  }
  try {
    parse(m);
  } catch (const char* e) {
    fprintf(stderr, "%s\n", e);
    return 1;
  }
  if (m.layers.empty()) return 1;

  // Plan once, before inference.
  Planner plan;
  std::vector<uint16_t> ids;
  plan_runner_graph(m, plan, ids);
  uint32_t largest = 0;
  for (uint16_t i = 0; i < plan.num_tensors(); ++i)
    if (plan.entry(i).size > largest) largest = plan.entry(i).size;

  printf("Activations (int8 bytes, planned offset):\n");
  printf("  input           %6u  @%u\n", plan.entry(ids[0]).size, plan.offset(ids[0]));
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const Layer& l = m.layers[i];
    const char* name = l.op == OP_CONV ? "conv" : (l.op == OP_POOL ? "maxpool" : "dense");
    printf("  %-7s %2ux%2ux%-3u %6u  @%u\n", name, l.oh, l.ow, l.oc,
           plan.entry(ids[i + 1]).size, plan.offset(ids[i + 1]));
  }

  Reader r = {test.data(), test.data() + test.size()};
  uint32_t count = r.get<uint32_t>();
  const uint16_t th = r.get<uint16_t>(), tw = r.get<uint16_t>(), tc = r.get<uint16_t>();
  const uint32_t pixels = uint32_t(th) * tw * tc;
  if (pixels != uint32_t(m.in_h) * m.in_w * m.in_c) {
    fprintf(stderr, "test set shape does not match the model\n");
    return 1;
  }
  const uint8_t* images = r.bytes(static_cast<size_t>(count) * pixels);
  const uint8_t* labels = r.bytes(count);
  if (count > limit) count = limit;

  // The planned arena is a single block from a RAM arena allocator, taken
  // once and kept for the life of the model.
  uint8_t* arena = static_cast<uint8_t*>(g_plan_ram.allocate(plan.arena_bytes()));
  if (!arena) return 1;
  const Placement planned = {&plan, ids.data(), arena};
  const Placement dynamic = {nullptr, nullptr, nullptr};
  g_ram.reset_high_water_mark();
  g_ram.rebalances = 0;

  const uint32_t n_out = m.layers.back().out_count();
  std::vector<int8_t> qin(pixels), out_dyn(n_out), out_plan(n_out);
  uint32_t correct = 0, mismatches = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* img = images + static_cast<size_t>(i) * pixels;
    for (uint32_t j = 0; j < pixels; ++j) {
      long q = lroundf((img[j] / 255.0f) / m.in_scale) + m.in_zp;
      qin[j] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    run(m, qin.data(), planned, out_plan.data());
    run(m, qin.data(), dynamic, out_dyn.data());
    if (memcmp(out_dyn.data(), out_plan.data(), n_out) != 0) ++mismatches;
    if (argmax(out_plan.data(), n_out) == labels[i]) ++correct;
    if (g_failed) {
      fprintf(stderr, "pTensor raised an error at image %u\n", i);
      return 1;
    }
  }

  Planner unfused, unfused_inplace;
  plan_unfused_graph(m, unfused, false);
  plan_unfused_graph(m, unfused_inplace, true);

  printf("\nPeak activation RAM, this network (%zu layers):\n", m.layers.size());
  printf("  one buffer per activation (naive)        %6u bytes\n", plan.naive_bytes());
  printf("  ping-pong, 2 x largest activation        %6u bytes\n", 2 * largest);
  printf("  dynamic RamTensors, arena high-water     %6zu bytes  (%u-byte arena, %.1f compactions per inference)\n",
         g_ram.high_water_mark(), (unsigned)DYNAMIC_ARENA_BYTES,
         count ? double(g_ram.rebalances) / count : 0.0);
  printf("  planned arena                            %6u bytes\n", plan.arena_bytes());
  printf("  lower bound, most bytes live at once     %6u bytes\n", plan.live_peak_bytes());
  printf("\nSame network with unfused ReLU/RESHAPE/SOFTMAX operators (planning only):\n");
  printf("  one buffer per activation (naive)        %6u bytes\n", unfused.naive_bytes());
  printf("  planned, no in-place                     %6u bytes\n", unfused.arena_bytes());
  printf("  planned, in-place where safe (%2u ops)    %6u bytes\n",
         unfused_inplace.inplace_count(), unfused_inplace.arena_bytes());
  printf("\nImages %u, accuracy %.2f%%, planned vs dynamic logits differ on %u\n",
         count, count ? 100.0 * correct / count : 0.0, mismatches);
  return mismatches ? 1 : 0;
}
