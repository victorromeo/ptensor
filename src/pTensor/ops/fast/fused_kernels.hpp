#ifndef PTENSOR_FAST_FUSED_KERNELS_H
#define PTENSOR_FAST_FUSED_KERNELS_H
/**
 * Kernels for the fused operators (ops/fused/FusedOperators.hpp).
 *
 * A fused kernel computes a conv / depthwise conv / fully connected / add and
 * then, while the value is still in a register, applies a per-channel
 * epilogue:
 *
 *     out[c] = act(acc * scale[c] + shift[c])     (scale given: folded BN)
 *     out[c] = act(acc + bias[c])                 (no scale)
 *
 * where act is a clamp to [lo, hi] (none, ReLU, ReLU6, ReLU1 or any range).
 * That replaces two or three passes over the output (conv, then batch norm,
 * then activation) with one, and the intermediate tensors go away.
 *
 * Without a scale, each kernel adds terms in exactly the order the unfused
 * operator does, so fusing only the activation is bit-identical to running
 * the operators one after another.
 *
 * Every kernel is a template over small accessor types, so the same loop
 * serves dense arrays (DenseIn / DenseOut, which inline to plain pointer
 * loads) and any other tensor (AnyIn / AnyOut, which fall back to the
 * virtual read, e.g. for PackedRomTensor weights). Nothing is emitted unless a
 * fused operator is instantiated.
 */
#include <limits>
#include <stdint.h>

#include "pTensor/core/tensor.hpp"
#include "pTensor/ops/fast/dense_kernels.hpp"

namespace pTensor {

/**
 * The activation a fused operator applies to its output, as a clamp.
 * TFLite's fused activations (NONE, RELU, RELU6, RELU_N1_TO_1) are all
 * clamps, so this covers everything a converter fuses.
 */
struct FusedActivation {
  float lo, hi;

  FusedActivation()
      : lo(-std::numeric_limits<float>::infinity()),
        hi(std::numeric_limits<float>::infinity()) {}
  FusedActivation(float lo_, float hi_) : lo(lo_), hi(hi_) {}

  static FusedActivation none() { return FusedActivation(); }
  static FusedActivation relu() {
    return FusedActivation(0.0f, std::numeric_limits<float>::infinity());
  }
  static FusedActivation relu6() { return FusedActivation(0.0f, 6.0f); }
  static FusedActivation relu1() { return FusedActivation(-1.0f, 1.0f); }
  static FusedActivation clamp(float lo_, float hi_) {
    return FusedActivation(lo_, hi_);
  }

  // Same comparisons as the standalone ReLU / ReLU6 operators, so results
  // (including the sign of zero) match them exactly. NaN passes through.
  inline float operator()(float x) const {
    return x < lo ? lo : (x > hi ? hi : x);
  }
};

namespace fused {

// ---- accessors ------------------------------------------------------------

template <typename T>
struct DenseIn {
  const T* p;
  explicit DenseIn(const T* p_ = nullptr) : p(p_) {}
  inline bool ok() const { return p != nullptr; }
  inline T operator[](uint32_t i) const { return p[i]; }
};

template <typename T>
struct DenseOut {
  T* p;
  explicit DenseOut(T* p_) : p(p_) {}
  inline void set(uint32_t i, T v) const { p[i] = v; }
};

/**
 * Reads a dense array when it has one and falls back to the tensor's virtual
 * read otherwise. Used for the non-dense path, where the per-element branch
 * costs little next to the virtual call (and to decoding packed weights).
 */
template <typename T>
struct AnyIn {
  const T* p;
  const Tensor* t;
  AnyIn() : p(nullptr), t(nullptr) {}
  AnyIn(const T* p_, const Tensor* t_) : p(p_), t(t_) {}
  inline bool ok() const { return p != nullptr || t != nullptr; }
  inline T operator[](uint32_t i) const {
    return p ? p[i] : static_cast<T>((*t)(i));
  }
};

template <typename T>
struct AnyOut {
  T* p;
  Tensor* t;
  AnyOut(T* p_, Tensor* t_) : p(p_), t(t_) {}
  inline void set(uint32_t i, T v) const {
    if (p)
      p[i] = v;
    else
      (*t)(i) = v;
  }
};

// out = act(acc * scale + shift) or act(acc + bias).
template <typename B, typename S>
inline float epilogue(float acc, const B& bias, const S& scale, uint32_t c,
                      const FusedActivation& act) {
  const float b = bias.ok() ? bias[c] : 0.0f;
  return act(scale.ok() ? acc * scale[c] + b : acc + b);
}

// ---- conv2d ---------------------------------------------------------------

/**
 * NHWC conv with an OHWI filter, fused epilogue. The accumulation order is
 * that of fast::conv2d_nhwc_ohwi (ky, kx, then input channel).
 */
template <typename In, typename F, typename B, typename S, typename Out>
inline void conv2d(const In& in, const F& f, const B& bias, const S& scale,
                   const Out& out, const fast::ConvGeometry& g,
                   const FusedActivation& act) {
  const int32_t C = g.in_c;
  for (int32_t n = 0; n < g.batches; ++n) {
    const uint32_t in_n = static_cast<uint32_t>(n * g.in_h * g.in_w * C);
    for (int32_t oy = 0; oy < g.out_h; ++oy) {
      const int32_t iy0 = oy * g.stride_h - g.pad_top;
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const int32_t ix0 = ox * g.stride_w - g.pad_left;
        const uint32_t o =
            static_cast<uint32_t>(((n * g.out_h + oy) * g.out_w + ox) * g.out_c);
        for (int32_t oc = 0; oc < g.out_c; ++oc) {
          const uint32_t f_oc = static_cast<uint32_t>(oc * g.k_h * g.k_w * C);
          float acc = 0;
          for (int32_t ky = 0; ky < g.k_h; ++ky) {
            const int32_t iy = iy0 + ky * g.dil_h;
            if (iy < 0 || iy >= g.in_h) continue;
            for (int32_t kx = 0; kx < g.k_w; ++kx) {
              const int32_t ix = ix0 + kx * g.dil_w;
              if (ix < 0 || ix >= g.in_w) continue;
              const uint32_t ip = in_n + static_cast<uint32_t>((iy * g.in_w + ix) * C);
              const uint32_t fp = f_oc + static_cast<uint32_t>((ky * g.k_w + kx) * C);
              for (int32_t c = 0; c < C; ++c) acc += in[ip + c] * f[fp + c];
            }
          }
          out.set(o + oc, epilogue(acc, bias, scale, oc, act));
        }
      }
    }
  }
}

/**
 * Dense-pointer version of conv2d() above (same sums, same order), laid out
 * like fast::conv2d_nhwc_ohwi so the compiler keeps everything in pointers.
 */
inline void conv2d_dense(const float* in, const float* filter,
                         const float* bias, const float* scale, float* out,
                         const fast::ConvGeometry& g,
                         const FusedActivation& act_ref) {
  // Local copies: the output stores could otherwise alias act / g, which
  // makes the compiler reload them inside the loops.
  const FusedActivation act = act_ref;
  const int32_t C = g.in_c;
  for (int32_t n = 0; n < g.batches; ++n) {
    const float* in_n = in + n * g.in_h * g.in_w * C;
    for (int32_t oy = 0; oy < g.out_h; ++oy) {
      const int32_t iy0 = oy * g.stride_h - g.pad_top;
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const int32_t ix0 = ox * g.stride_w - g.pad_left;
        float* o = out + ((n * g.out_h + oy) * g.out_w + ox) * g.out_c;
        for (int32_t oc = 0; oc < g.out_c; ++oc) {
          const float* f_oc = filter + oc * g.k_h * g.k_w * C;
          float acc = 0;
          for (int32_t ky = 0; ky < g.k_h; ++ky) {
            const int32_t iy = iy0 + ky * g.dil_h;
            if (iy < 0 || iy >= g.in_h) continue;
            for (int32_t kx = 0; kx < g.k_w; ++kx) {
              const int32_t ix = ix0 + kx * g.dil_w;
              if (ix < 0 || ix >= g.in_w) continue;
              const float* ip = in_n + (iy * g.in_w + ix) * C;
              const float* fp = f_oc + (ky * g.k_w + kx) * C;
              for (int32_t c = 0; c < C; ++c) acc += ip[c] * fp[c];
            }
          }
          o[oc] = acc;
        }
        // Epilogue over the (cache-hot) output pixel, kept out of the
        // accumulation loop so that loop compiles exactly like the plain one.
        if (scale && bias) {
          for (int32_t oc = 0; oc < g.out_c; ++oc) o[oc] = act(o[oc] * scale[oc] + bias[oc]);
        } else if (scale) {
          for (int32_t oc = 0; oc < g.out_c; ++oc) o[oc] = act(o[oc] * scale[oc] + 0.0f);
        } else if (bias) {
          for (int32_t oc = 0; oc < g.out_c; ++oc) o[oc] = act(o[oc] + bias[oc]);
        } else {
          for (int32_t oc = 0; oc < g.out_c; ++oc) o[oc] = act(o[oc] + 0.0f);
        }
      }
    }
  }
}

// ---- depthwise conv2d -----------------------------------------------------

/**
 * Dense NHWC depthwise conv, filter [1, k_h, k_w, in_c * mult]. Accumulates a
 * whole output pixel in place (the layout of fast::depthwise_conv2d_nhwc),
 * then applies the epilogue to that row while it is still in cache. Without
 * a scale the accumulator starts at the bias, as the unfused operator does.
 */
inline void depthwise_conv2d_dense(const float* in, const float* filter,
                                   const float* bias, const float* scale,
                                   float* out, const fast::ConvGeometry& g,
                                   int32_t mult,
                                   const FusedActivation& act_ref) {
  const FusedActivation act = act_ref;  // local: see conv2d_dense
  const int32_t C = g.in_c, OC = g.out_c;
  for (int32_t n = 0; n < g.batches; ++n) {
    const float* in_n = in + n * g.in_h * g.in_w * C;
    for (int32_t oy = 0; oy < g.out_h; ++oy) {
      const int32_t iy0 = oy * g.stride_h - g.pad_top;
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const int32_t ix0 = ox * g.stride_w - g.pad_left;
        float* acc = out + ((n * g.out_h + oy) * g.out_w + ox) * OC;
        if (scale || !bias) {
          for (int32_t oc = 0; oc < OC; ++oc) acc[oc] = 0.0f;
        } else {
          for (int32_t oc = 0; oc < OC; ++oc) acc[oc] = bias[oc];
        }
        for (int32_t ky = 0; ky < g.k_h; ++ky) {
          const int32_t iy = iy0 + ky * g.dil_h;
          if (iy < 0 || iy >= g.in_h) continue;
          for (int32_t kx = 0; kx < g.k_w; ++kx) {
            const int32_t ix = ix0 + kx * g.dil_w;
            if (ix < 0 || ix >= g.in_w) continue;
            const float* ip = in_n + (iy * g.in_w + ix) * C;
            const float* fp = filter + (ky * g.k_w + kx) * OC;
            if (mult == 1) {
              for (int32_t c = 0; c < C; ++c) acc[c] += ip[c] * fp[c];
            } else {
              for (int32_t c = 0; c < C; ++c) {
                const float xv = ip[c];
                for (int32_t m = 0; m < mult; ++m)
                  acc[c * mult + m] += xv * fp[c * mult + m];
              }
            }
          }
        }
        if (scale) {
          if (bias) {
            for (int32_t oc = 0; oc < OC; ++oc)
              acc[oc] = act(acc[oc] * scale[oc] + bias[oc]);
          } else {
            for (int32_t oc = 0; oc < OC; ++oc)
              acc[oc] = act(acc[oc] * scale[oc]);
          }
        } else {
          for (int32_t oc = 0; oc < OC; ++oc) acc[oc] = act(acc[oc]);
        }
      }
    }
  }
}

/** Depthwise conv through accessors (any tensor); same sums as the dense one. */
template <typename In, typename F, typename B, typename S, typename Out>
inline void depthwise_conv2d(const In& in, const F& f, const B& bias,
                             const S& scale, const Out& out,
                             const fast::ConvGeometry& g, int32_t mult,
                             const FusedActivation& act) {
  const int32_t C = g.in_c, OC = g.out_c;
  for (int32_t n = 0; n < g.batches; ++n)
    for (int32_t oy = 0; oy < g.out_h; ++oy)
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const uint32_t o =
            static_cast<uint32_t>(((n * g.out_h + oy) * g.out_w + ox) * OC);
        for (int32_t oc = 0; oc < OC; ++oc) {
          const int32_t ic = oc / mult;
          float acc = (!scale.ok() && bias.ok()) ? bias[oc] : 0.0f;
          for (int32_t ky = 0; ky < g.k_h; ++ky) {
            const int32_t iy = oy * g.stride_h - g.pad_top + ky * g.dil_h;
            if (iy < 0 || iy >= g.in_h) continue;
            for (int32_t kx = 0; kx < g.k_w; ++kx) {
              const int32_t ix = ox * g.stride_w - g.pad_left + kx * g.dil_w;
              if (ix < 0 || ix >= g.in_w) continue;
              acc += in[static_cast<uint32_t>(((n * g.in_h + iy) * g.in_w + ix) * C + ic)] *
                     f[static_cast<uint32_t>((ky * g.k_w + kx) * OC + oc)];
            }
          }
          float v = acc;
          if (scale.ok()) {
            v = acc * scale[oc];
            if (bias.ok()) v += bias[oc];
          }
          out.set(o + oc, act(v));
        }
      }
}

// ---- fully connected ------------------------------------------------------

/**
 * out[b, o] = epilogue(sum_d in[b, d] * w[d, o]), weights [in, out]. Dense
 * case runs fast::matmul (same per-element sum order) and then the epilogue
 * over each output row.
 */
inline void fully_connected_dense(const float* in, const float* w,
                                  const float* bias, const float* scale,
                                  float* out, uint32_t batches, uint32_t depth,
                                  uint32_t units,
                                  const FusedActivation& act_ref) {
  const FusedActivation act = act_ref;  // local: see conv2d_dense
  fast::matmul(in, w, out, batches, depth, units);
  for (uint32_t b = 0; b < batches; ++b) {
    float* row = out + b * units;
    if (scale) {
      if (bias) {
        for (uint32_t o = 0; o < units; ++o) row[o] = act(row[o] * scale[o] + bias[o]);
      } else {
        for (uint32_t o = 0; o < units; ++o) row[o] = act(row[o] * scale[o] + 0.0f);
      }
    } else if (bias) {
      for (uint32_t o = 0; o < units; ++o) row[o] = act(row[o] + bias[o]);
    } else {
      // "+ 0" as the unfused operator does (it turns -0 into +0).
      for (uint32_t o = 0; o < units; ++o) row[o] = act(row[o] + 0.0f);
    }
  }
}

template <typename In, typename F, typename B, typename S, typename Out>
inline void fully_connected(const In& in, const F& w, const B& bias,
                            const S& scale, const Out& out, uint32_t batches,
                            uint32_t depth, uint32_t units,
                            const FusedActivation& act) {
  for (uint32_t b = 0; b < batches; ++b)
    for (uint32_t o = 0; o < units; ++o) {
      float acc = 0;
      for (uint32_t d = 0; d < depth; ++d) acc += in[b * depth + d] * w[d * units + o];
      out.set(b * units + o, epilogue(acc, bias, scale, o, act));
    }
}

// ---- add ------------------------------------------------------------------

/** c[i] = act(a[i] + b[i]) over n elements (no broadcasting). */
inline void add_dense(const float* a, const float* b, float* c, uint32_t n,
                      const FusedActivation& act_ref) {
  // Clamp bounds in registers, not behind a pointer the stores might alias,
  // so the loop vectorises.
  const float lo = act_ref.lo, hi = act_ref.hi;
  for (uint32_t i = 0; i < n; ++i) {
    const float v = a[i] + b[i];
    c[i] = v < lo ? lo : (v > hi ? hi : v);
  }
}

/** act(a + b): the functor broadcast_zip runs for FusedAddOperator. */
struct AddAct {
  FusedActivation act;
  explicit AddAct(const FusedActivation& a) : act(a) {}
  inline float operator()(float x, float y) const { return act(x + y); }
};

}  // namespace fused
}  // namespace pTensor
#endif
