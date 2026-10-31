#ifndef PTENSOR_FAST_DENSE_KERNELS_H
#define PTENSOR_FAST_DENSE_KERNELS_H
/**
 * Pointer-based kernels used by the reference operators when every tensor
 * involved is a dense array (see core/fast_access.hpp). They compute exactly
 * what the reference loops compute, in the same accumulation order, so
 * results are bit-identical; they only drop the per-element virtual call and
 * index arithmetic, and lay loops out so the inner one walks contiguous
 * memory (which compilers auto-vectorise on hosts and keeps Cortex-M loads
 * sequential).
 *
 * All functions are templates: nothing is emitted unless an operator that
 * uses them is instantiated.
 */
#include <stdint.h>

namespace pTensor {
namespace fast {

/** c[M,N] = a[M,K] @ b[K,N]. Accumulates each row of c in place. */
template <typename T>
inline void matmul(const T* a, const T* b, T* c, uint32_t M, uint32_t K,
                   uint32_t N) {
  for (uint32_t i = 0; i < M; ++i) {
    T* crow = c + i * N;
    for (uint32_t j = 0; j < N; ++j) crow[j] = 0;
    const T* arow = a + i * K;
    for (uint32_t k = 0; k < K; ++k) {
      const T aik = arow[k];
      const T* brow = b + k * N;
      for (uint32_t j = 0; j < N; ++j) crow[j] += aik * brow[j];
    }
  }
}

/** Elementwise out[i] = fn(in[i]). */
template <typename T, typename Fn>
inline void map(const T* in, T* out, uint32_t n, const Fn& fn) {
  for (uint32_t i = 0; i < n; ++i) out[i] = fn(in[i]);
}

/** Elementwise c[i] = fn(a[i], b[i]). */
template <typename T, typename Fn>
inline void zip(const T* a, const T* b, T* c, uint32_t n, const Fn& fn) {
  for (uint32_t i = 0; i < n; ++i) c[i] = fn(a[i], b[i]);
}

/**
 * 4D broadcast c = fn(a, b). dims are the output dims, sa/sb the input
 * strides per output dim (0 where broadcast). Specialises the innermost loop
 * for the common stride patterns.
 */
template <typename T, typename Fn>
inline void broadcast_zip(const T* a, const uint32_t (&sa)[4], const T* b,
                          const uint32_t (&sb)[4], T* c,
                          const uint32_t (&dims)[4], const Fn& fn) {
  const uint32_t n3 = dims[3];
  for (uint32_t i0 = 0; i0 < dims[0]; ++i0)
    for (uint32_t i1 = 0; i1 < dims[1]; ++i1)
      for (uint32_t i2 = 0; i2 < dims[2]; ++i2) {
        const T* pa = a + i0 * sa[0] + i1 * sa[1] + i2 * sa[2];
        const T* pb = b + i0 * sb[0] + i1 * sb[1] + i2 * sb[2];
        if (sa[3] == 1 && sb[3] == 1) {
          for (uint32_t i = 0; i < n3; ++i) c[i] = fn(pa[i], pb[i]);
        } else if (sa[3] == 1 && sb[3] == 0) {
          const T y = *pb;
          for (uint32_t i = 0; i < n3; ++i) c[i] = fn(pa[i], y);
        } else if (sa[3] == 0 && sb[3] == 1) {
          const T x = *pa;
          for (uint32_t i = 0; i < n3; ++i) c[i] = fn(x, pb[i]);
        } else {
          for (uint32_t i = 0; i < n3; ++i) c[i] = fn(pa[i * sa[3]], pb[i * sb[3]]);
        }
        c += n3;
      }
}

/** Geometry shared by the conv kernels. Offsets are subtracted from out*stride. */
struct ConvGeometry {
  int32_t batches, in_h, in_w, in_c;
  int32_t out_h, out_w, out_c;
  int32_t k_h, k_w;
  int32_t stride_h, stride_w;
  int32_t dil_h, dil_w;
  int32_t pad_top, pad_left;
};

/**
 * NHWC conv with an OHWI filter [out_c, k_h, k_w, in_c]; bias may be null.
 * Per output: acc = sum over (ky, kx, ci) in that order, then out = acc + bias,
 * matching generic_convolution_kernel.
 */
template <typename T>
inline void conv2d_nhwc_ohwi(const T* in, const T* filter, const T* bias,
                             T* out, const ConvGeometry& g) {
  const int32_t C = g.in_c;
  for (int32_t n = 0; n < g.batches; ++n) {
    const T* in_n = in + n * g.in_h * g.in_w * C;
    for (int32_t oy = 0; oy < g.out_h; ++oy) {
      const int32_t iy0 = oy * g.stride_h - g.pad_top;
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const int32_t ix0 = ox * g.stride_w - g.pad_left;
        T* o = out + ((n * g.out_h + oy) * g.out_w + ox) * g.out_c;
        for (int32_t oc = 0; oc < g.out_c; ++oc) {
          const T* f_oc = filter + oc * g.k_h * g.k_w * C;
          T acc = 0;
          for (int32_t ky = 0; ky < g.k_h; ++ky) {
            const int32_t iy = iy0 + ky * g.dil_h;
            if (iy < 0 || iy >= g.in_h) continue;
            for (int32_t kx = 0; kx < g.k_w; ++kx) {
              const int32_t ix = ix0 + kx * g.dil_w;
              if (ix < 0 || ix >= g.in_w) continue;
              const T* ip = in_n + (iy * g.in_w + ix) * C;
              const T* fp = f_oc + (ky * g.k_w + kx) * C;
              for (int32_t c = 0; c < C; ++c) acc += ip[c] * fp[c];
            }
          }
          o[oc] = static_cast<T>(acc + (bias ? bias[oc] : T(0)));
        }
      }
    }
  }
}

/**
 * NHWC depthwise conv, filter [1, k_h, k_w, in_c * mult], out_c = in_c * mult.
 * Per output: acc = bias, then += over (ky, kx), as DepthwiseConv2dOperator.
 * Accumulates directly in the output row, so T is the accumulator type.
 */
template <typename T>
inline void depthwise_conv2d_nhwc(const T* in, const T* filter, const T* bias,
                                  T* out, const ConvGeometry& g, int32_t mult) {
  const int32_t C = g.in_c, OC = g.out_c;
  for (int32_t n = 0; n < g.batches; ++n) {
    const T* in_n = in + n * g.in_h * g.in_w * C;
    for (int32_t oy = 0; oy < g.out_h; ++oy) {
      const int32_t iy0 = oy * g.stride_h - g.pad_top;
      for (int32_t ox = 0; ox < g.out_w; ++ox) {
        const int32_t ix0 = ox * g.stride_w - g.pad_left;
        T* acc = out + ((n * g.out_h + oy) * g.out_w + ox) * OC;
        for (int32_t oc = 0; oc < OC; ++oc) acc[oc] = bias ? bias[oc] : T(0);
        for (int32_t ky = 0; ky < g.k_h; ++ky) {
          const int32_t iy = iy0 + ky * g.dil_h;
          if (iy < 0 || iy >= g.in_h) continue;
          for (int32_t kx = 0; kx < g.k_w; ++kx) {
            const int32_t ix = ix0 + kx * g.dil_w;
            if (ix < 0 || ix >= g.in_w) continue;
            const T* ip = in_n + (iy * g.in_w + ix) * C;
            const T* fp = filter + (ky * g.k_w + kx) * OC;
            if (mult == 1) {
              for (int32_t c = 0; c < C; ++c) acc[c] += ip[c] * fp[c];
            } else {
              for (int32_t c = 0; c < C; ++c) {
                const T xv = ip[c];
                for (int32_t m = 0; m < mult; ++m)
                  acc[c * mult + m] += xv * fp[c * mult + m];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace fast
}  // namespace pTensor
#endif
