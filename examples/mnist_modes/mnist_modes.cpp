// Runs a .ptm model (written by python/ptensor/ptm.py) over a test set with
// pTensor and reports accuracy, inference time and weight storage.
//
//   mnist_modes model.ptm test.bin [--ram] [--limit N]
//
// Weights stay in the storage mode the file chose: float32 weights become
// RomTensors, packed weights PackedRomTensors and Hough weights
// HoughRomTensors, decoded on the fly as operators read them (as on a
// microcontroller running from flash). --ram decodes packed and Hough weights
// to float32 once at load instead, which trades RAM for the dense fast paths.
//
// test.bin: u32 count, u16 h, u16 w, u16 c, count*h*w*c u8 pixels, count u8 labels.
// Output: one JSON object on stdout.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pTensor.h"

using namespace pTensor;
using namespace pTensor::ReferenceOperators;

namespace {

localCircularArenaAllocator<16 * 1024, uint32_t, 128> g_meta;
localCircularArenaAllocator<32 * 1024, uint32_t, 64> g_ram;
SimpleErrorHandler g_err(4);
bool g_failed = false;

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

enum Storage { ST_F32 = 0, ST_PACKED = 1, ST_HOUGH = 2, ST_I8 = 3, ST_I32 = 4 };

struct TensorRec {
  uint8_t storage = 0;
  std::vector<uint16_t> dims;
  // packed
  uint8_t format = 0, bits = 0;
  float scale = 1.0f;
  int32_t zero_point = 0;
  // hough
  uint8_t line_bits = 0, t_bits = 0;
  uint16_t stride = 1;
  std::vector<float> lines;
  uint32_t nbytes = 0;
  std::vector<uint32_t> aligned;  // 4-byte aligned copy of the payload
  const uint8_t* data() const {
    return reinterpret_cast<const uint8_t*>(aligned.data());
  }
  std::vector<float> decoded;  // --ram

  uint32_t count() const {
    uint32_t n = 1;
    for (uint16_t d : dims) n *= d;
    return n;
  }
  size_t storage_bytes() const {
    size_t extra = 0;
    if (storage == ST_PACKED && format != (uint8_t)PackedFormat::Float16 &&
        format != (uint8_t)PackedFormat::Float32 &&
        format != (uint8_t)PackedFormat::Float64)
      extra = 8;  // scale + zero point
    if (storage == ST_HOUGH) extra = lines.size() * 4 + 4;
    return nbytes + extra;
  }
  TensorShape shape() const {
    switch (dims.size()) {
      case 1: return TensorShape(dims[0]);
      case 2: return TensorShape(dims[0], dims[1]);
      case 3: return TensorShape(dims[0], dims[1], dims[2]);
      default: return TensorShape(dims[0], dims[1], dims[2], dims[3]);
    }
  }
  PackedEncoding packed() const {
    PackedEncoding e = PackedEncoding::of((PackedFormat)format, scale, zero_point);
    e.bit_width = bits;
    return e;
  }
  HoughEncoding hough() const {
    return HoughEncoding::make(line_bits, t_bits, stride, lines.data());
  }
  void decode_to_ram() {
    if (storage == ST_PACKED) {
      decoded.resize(count());
      Tensor t = new PackedRomTensor(shape(), flt, packed(), data());
      const Tensor& ct = t;
      for (uint32_t i = 0; i < count(); ++i) decoded[i] = static_cast<float>(ct(i));
    } else if (storage == ST_HOUGH) {
      decoded.resize(count());
      hough::decode_all(data(), lines.data(), line_bits, t_bits, stride, count(),
                        decoded.data());
    }
  }
  Tensor make() const {
    if (!decoded.empty()) return new RomTensor(shape(), flt, decoded.data());
    switch (storage) {
      case ST_F32: return new RomTensor(shape(), flt, data());
      case ST_I8: return new RomTensor(shape(), i8, data());
      case ST_I32: return new RomTensor(shape(), i32, data());
      case ST_PACKED: return new PackedRomTensor(shape(), flt, packed(), data());
      default: return new HoughRomTensor(shape(), hough(), data());
    }
  }
};

TensorRec read_tensor(Reader& r) {
  TensorRec t;
  t.storage = r.get<uint8_t>();
  const uint8_t rank = r.get<uint8_t>();
  for (uint8_t i = 0; i < rank; ++i) t.dims.push_back(r.get<uint16_t>());
  if (t.storage == ST_PACKED) {
    t.format = r.get<uint8_t>();
    t.bits = r.get<uint8_t>();
    t.scale = r.get<float>();
    t.zero_point = r.get<int32_t>();
  } else if (t.storage == ST_HOUGH) {
    t.line_bits = r.get<uint8_t>();
    t.t_bits = r.get<uint8_t>();
    t.stride = r.get<uint16_t>();
    t.lines.resize(4u << t.line_bits);
    memcpy(t.lines.data(), r.bytes(t.lines.size() * 4), t.lines.size() * 4);
  }
  t.nbytes = r.get<uint32_t>();
  // The file packs records back to back; float/int32 kernels need aligned data.
  t.aligned.resize((t.nbytes + 3) / 4);
  memcpy(t.aligned.data(), r.bytes(t.nbytes), t.nbytes);
  return t;
}

enum Op { OP_CONV = 1, OP_POOL = 2, OP_DENSE = 3 };

struct Layer {
  uint8_t op = 0, act = 0, padding = 2, kh = 0, kw = 0, sh = 1, sw = 1;
  float out_scale = 1.0f;
  int32_t out_zp = 0;
  std::vector<float> w_scales;
  std::vector<int32_t> w_zps;
  TensorRec w, b;
};

struct Model {
  std::vector<uint8_t> file;
  uint8_t kind = 0;  // 0 float, 1 int8
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

void parse(Model& m) {
  Reader r = {m.file.data(), m.file.data() + m.file.size()};
  if (memcmp(r.bytes(4), "PTM1", 4) != 0) throw "not a .ptm file";
  m.kind = r.get<uint8_t>();
  m.in_h = r.get<uint16_t>();
  m.in_w = r.get<uint16_t>();
  m.in_c = r.get<uint16_t>();
  m.in_scale = r.get<float>();
  m.in_zp = r.get<int32_t>();
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
      if (m.kind == 1) {
        l.out_scale = r.get<float>();
        l.out_zp = r.get<int32_t>();
        const uint32_t ns = r.get<uint32_t>();
        for (uint32_t j = 0; j < ns; ++j) l.w_scales.push_back(r.get<float>());
        l.w_zps.assign(ns, 0);
      }
      l.w = read_tensor(r);
      l.b = read_tensor(r);
    }
    m.layers.push_back(l);
  }
}

uint16_t out_size(uint16_t in, uint16_t k, uint16_t s, uint8_t padding) {
  if (padding == SAME) return (in + s - 1) / s;
  return (in - k) / s + 1;
}

// ---- float network --------------------------------------------------------

void run_float(const Model& m, const float* input, std::vector<float>& a,
               std::vector<float>& b, uint32_t* out_count) {
  uint16_t h = m.in_h, w = m.in_w, c = m.in_c;
  memcpy(a.data(), input, sizeof(float) * h * w * c);
  bool flat = false;
  uint32_t n = h * w * c;
  for (const Layer& l : m.layers) {
    if (l.op == OP_CONV) {
      const uint16_t oc = l.w.dims[0], k_h = l.w.dims[1], k_w = l.w.dims[2];
      const uint16_t oh = out_size(h, k_h, l.sh, l.padding);
      const uint16_t ow = out_size(w, k_w, l.sw, l.padding);
      {
        typedef Conv2dOperator<float> Op;
        Tensor x = new RomTensor({1, h, w, c}, flt, a.data());
        Tensor f = l.w.make();
        Tensor bias = l.b.make();
        Tensor y = new BufferTensor({1, oh, ow, oc}, flt, b.data());
        Op op({l.sh, l.sw}, static_cast<Padding>(l.padding));
        op.set_inputs({{Op::in, x}, {Op::filter, f}, {Op::bias, bias}})
            .set_outputs({{Op::out, y}}).eval();
      }
      h = oh; w = ow; c = oc;
      n = h * w * c;
      if (l.act == 1)
        for (uint32_t i = 0; i < n; ++i) b[i] = b[i] > 0 ? b[i] : 0;
      else if (l.act == 3)
        for (uint32_t i = 0; i < n; ++i) b[i] = b[i] < 0 ? 0 : (b[i] > 6 ? 6 : b[i]);
    } else if (l.op == OP_POOL) {
      const uint16_t oh = out_size(h, l.kh, l.sh, l.padding);
      const uint16_t ow = out_size(w, l.kw, l.sw, l.padding);
      {
        typedef MaxPoolOperator<float> Op;
        Tensor x = new RomTensor({1, h, w, c}, flt, a.data());
        Tensor y = new BufferTensor({1, oh, ow, c}, flt, b.data());
        Op op({l.kh, l.kw}, {1, l.sh, l.sw, 1}, static_cast<Padding>(l.padding));
        op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
      }
      h = oh; w = ow;
      n = h * w * c;
    } else {
      const uint16_t in_f = l.w.dims[0], out_f = l.w.dims[1];
      (void)flat;
      {
        typedef FullyConnectedOperator<float> FC;
        Tensor x = new RomTensor({1, in_f}, flt, a.data());
        Tensor wt = l.w.make();
        Tensor bias = l.b.make();
        Tensor y = new BufferTensor({1, out_f}, flt, b.data());
        FC op(l.act == 1 ? Fuseable::ReLU<float>
                         : (l.act == 3 ? Fuseable::ReLU6<float> : Fuseable::NoActivation<float>));
        op.set_inputs({{FC::input, x}, {FC::filter, wt}, {FC::bias, bias}})
            .set_outputs({{FC::output, y}}).eval();
      }
      n = out_f;
      flat = true;
    }
    a.swap(b);
  }
  *out_count = n;
}

// ---- int8 network ---------------------------------------------------------

void run_int8(const Model& m, const int8_t* input, std::vector<int8_t>& a,
              std::vector<int8_t>& b, uint32_t* out_count) {
  uint16_t h = m.in_h, w = m.in_w, c = m.in_c;
  memcpy(a.data(), input, h * w * c);
  float in_scale = m.in_scale;
  int32_t in_zp = m.in_zp;
  uint32_t n = h * w * c;
  for (const Layer& l : m.layers) {
    if (l.op == OP_CONV) {
      const uint16_t oc = l.w.dims[0], k_h = l.w.dims[1], k_w = l.w.dims[2];
      const uint16_t oh = out_size(h, k_h, l.sh, l.padding);
      const uint16_t ow = out_size(w, k_w, l.sw, l.padding);
      {
        typedef TflmSymQuantOps::Conv2dOperator<int8_t> Op;
        Tensor x = new RomTensor({1, h, w, c}, i8, a.data());
        x->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
        Tensor f = l.w.make();
        f->set_quantization_params(PerChannelQuantizationParams(
            l.w_zps.data(), l.w_scales.data(), (int)l.w_scales.size()));
        Tensor bias = l.b.make();
        Tensor y = new BufferTensor({1, oh, ow, oc}, i8, b.data());
        y->set_quantization_params(PerTensorQuantizationParams(l.out_zp, l.out_scale));
        const uint16_t strides[2] = {l.sh, l.sw};
        const uint16_t dilation[2] = {1, 1};
        Op op(strides, static_cast<Padding>(l.padding), dilation,
              static_cast<TFLM::TfLiteFusedActivation>(l.act));
        op.set_inputs({{Op::in, x}, {Op::filter, f}, {Op::bias, bias}})
            .set_outputs({{Op::out, y}}).eval();
      }
      h = oh; w = ow; c = oc;
      n = h * w * c;
      in_scale = l.out_scale;
      in_zp = l.out_zp;
    } else if (l.op == OP_POOL) {
      const uint16_t oh = out_size(h, l.kh, l.sh, l.padding);
      const uint16_t ow = out_size(w, l.kw, l.sw, l.padding);
      {
        typedef MaxPoolOperator<int8_t> Op;
        Tensor x = new RomTensor({1, h, w, c}, i8, a.data());
        Tensor y = new BufferTensor({1, oh, ow, c}, i8, b.data());
        Op op({l.kh, l.kw}, {1, l.sh, l.sw, 1}, static_cast<Padding>(l.padding));
        op.set_inputs({{Op::in, x}}).set_outputs({{Op::out, y}}).eval();
      }
      h = oh; w = ow;
      n = h * w * c;
    } else {
      const uint16_t in_f = l.w.dims[0], out_f = l.w.dims[1];
      {
        typedef TflmSymQuantOps::FullyConnectedOperator<int8_t> FC;
        Tensor x = new RomTensor({1, in_f}, i8, a.data());
        x->set_quantization_params(PerTensorQuantizationParams(in_zp, in_scale));
        Tensor wt = l.w.make();
        wt->set_quantization_params(PerChannelQuantizationParams(
            l.w_zps.data(), l.w_scales.data(), (int)l.w_scales.size()));
        Tensor bias = l.b.make();
        Tensor y = new BufferTensor({1, out_f}, i8, b.data());
        y->set_quantization_params(PerTensorQuantizationParams(l.out_zp, l.out_scale));
        FC op(static_cast<TFLM::TfLiteFusedActivation>(l.act));
        op.set_inputs({{FC::input, x}, {FC::filter, wt}, {FC::bias, bias}})
            .set_outputs({{FC::output, y}}).eval();
      }
      n = out_f;
      in_scale = l.out_scale;
      in_zp = l.out_zp;
    }
    a.swap(b);
  }
  *out_count = n;
}

template <typename T>
uint32_t argmax(const std::vector<T>& v, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.ptm test.bin [--ram] [--limit N]\n", argv[0]);
    return 2;
  }
  bool ram = false;
  uint32_t limit = 0xFFFFFFFFu;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--ram")) ram = true;
    else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
  }
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
  size_t weight_bytes = 0;
  for (Layer& l : m.layers) {
    if (l.op == OP_POOL) continue;
    weight_bytes += l.w.storage_bytes() + l.b.storage_bytes();
    if (m.kind == 1) weight_bytes += 4 * l.w_scales.size() + 8;
    if (ram) {
      l.w.decode_to_ram();
      l.b.decode_to_ram();
    }
  }

  Reader r = {test.data(), test.data() + test.size()};
  uint32_t count = r.get<uint32_t>();
  const uint16_t th = r.get<uint16_t>(), tw = r.get<uint16_t>(), tc = r.get<uint16_t>();
  const uint32_t pixels = th * tw * tc;
  const uint8_t* images = r.bytes(static_cast<size_t>(count) * pixels);
  const uint8_t* labels = r.bytes(count);
  if (count > limit) count = limit;

  const size_t act = 64 * 1024;
  std::vector<float> fa(act), fb(act), fin(pixels);
  std::vector<int8_t> qa(act), qb(act), qin(pixels);
  uint32_t correct = 0, n_out = 0;
  double total_us = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* img = images + static_cast<size_t>(i) * pixels;
    uint32_t pred;
    auto t0 = std::chrono::steady_clock::now();
    if (m.kind == 0) {
      for (uint32_t j = 0; j < pixels; ++j) fin[j] = img[j] / 255.0f;
      run_float(m, fin.data(), fa, fb, &n_out);
      pred = argmax(fa, n_out);
    } else {
      for (uint32_t j = 0; j < pixels; ++j) {
        long q = lroundf((img[j] / 255.0f) / m.in_scale) + m.in_zp;
        qin[j] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
      run_int8(m, qin.data(), qa, qb, &n_out);
      pred = argmax(qa, n_out);
    }
    auto t1 = std::chrono::steady_clock::now();
    total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (pred == labels[i]) ++correct;
    if (g_failed) {
      fprintf(stderr, "pTensor raised an error at image %u\n", i);
      return 1;
    }
  }
  printf("{\"images\": %u, \"accuracy\": %.6f, \"us_per_image\": %.2f, "
         "\"weight_bytes\": %zu, \"file_bytes\": %zu, \"ram_decode\": %s}\n",
         count, count ? double(correct) / count : 0.0, count ? total_us / count : 0.0,
         weight_bytes, m.file.size(), ram ? "true" : "false");
  return 0;
}
