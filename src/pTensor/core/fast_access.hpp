#ifndef PTENSOR_FAST_ACCESS_H
#define PTENSOR_FAST_ACCESS_H
/**
 * Helpers that let an operator skip per-element virtual reads when its
 * tensors are plain dense arrays of the expected type.
 *
 *   const float* x = fast_read<float>(inputs[in].tensor());
 *   float* y = fast_write<float>(outputs[out].tensor());
 *   if (x && y) { ...pointer loop... } else { ...reference loop... }
 *
 * Both return nullptr when the tensor is not dense (PackedRomTensor, custom
 * tensors), is read-only (fast_write on a RomTensor), holds another element
 * type, or when fast kernels are disabled with PTENSOR_WITH_FAST_KERNELS=OFF
 * (then the fast branch is dead code and the compiler drops it).
 */
#include "pTensor/ptensor_config.h"
#include "tensor.hpp"

namespace pTensor {

template <typename T>
inline const T* fast_read(const Tensor& t) {
#if PTENSOR_WITH_FAST_KERNELS
  if (t->get_type() != ttype_from<T>::type) return nullptr;
  return static_cast<const T*>(t->raw_data());
#else
  (void)t;
  return nullptr;
#endif
}

template <typename T>
inline T* fast_write(Tensor& t) {
#if PTENSOR_WITH_FAST_KERNELS
  if (t->get_type() != ttype_from<T>::type) return nullptr;
  return static_cast<T*>(t->raw_mutable_data());
#else
  (void)t;
  return nullptr;
#endif
}

}  // namespace pTensor
#endif
