#ifndef PTENSOR_QUANT_PRIM_H
#define PTENSOR_QUANT_PRIM_H
#include "types.hpp"
#include "errorHandler.hpp"
#include "memoryManagementInterface.hpp"

namespace pTensor {

DECLARE_ERROR(AttemptToUseUnSpecializedQuantizeParamsError);

class QuantizationParams {
 public:
  QuantizationParams();
  QuantizationParams(const int32_t* zp, const float* scale, int num_channels);
  // TODO Move to cpp file
  QuantizationParams(const QuantizationParams&) = default;
  QuantizationParams& operator=(const QuantizationParams&) = default;
  QuantizationParams(QuantizationParams&&) = default;
  QuantizationParams& operator=(QuantizationParams&&) = default;

  virtual ~QuantizationParams();
  virtual int32_t get_zeroP_for_channel(int i) const;
  virtual float get_scale_for_channel(int i) const;
  inline const int num_channels() const { return _num_channels; };
  
  // Allocate the tensor metadata on a different heap from the data scratch pads
  // Note: as long as derived classes dont override new and delete, these will
  // get called correctly
  void* operator new(size_t sz);
  void operator delete(void* p);

 protected:
  const int32_t* _zp;
  const float* _scale;
  int _num_channels;
};

using NotQuantized = QuantizationParams;

inline bool is_quantized(const QuantizationParams& params) {
  return params.num_channels() > 0;
}
inline bool is_per_channel_quantization(const QuantizationParams& params) {
  return params.num_channels() > 1;
}

// Keep these types small: they are allocated in the metadata arena.
//
// Per-tensor parameters are stored by value. Storing pointers here (as the
// per-channel variant does) would dangle whenever the caller passes a literal,
// a temporary, or an `int` on targets where int32_t is `long`.
class PerTensorQuantizationParams : public QuantizationParams {
 public:
  PerTensorQuantizationParams(int32_t zp, float scale);
  PerTensorQuantizationParams(const PerTensorQuantizationParams& that);
  PerTensorQuantizationParams& operator=(const PerTensorQuantizationParams& that);
  virtual int32_t get_zeroP_for_channel(int i) const override;
  virtual float get_scale_for_channel(int i) const override;

 private:
  int32_t _zp_value;
  float _scale_value;
};

class PerChannelQuantizationParams : public QuantizationParams {
 public:
  // PerChannelQuantizationParams(const int32_t* zp, const int32_t* scale, int
  // num_channels) : _zp(zp), _scale(scale), _num_channels(num_channels) {}

  // Smarter compile time thing might be
  template <int num_channels>
  PerChannelQuantizationParams(const int32_t (&zp)[num_channels],
                               const float (&scale)[num_channels])
      : QuantizationParams(zp, scale, num_channels) {}

  // Runtime-sized variant. The arrays are borrowed, not copied: they must
  // outlive every tensor that uses these parameters.
  PerChannelQuantizationParams(const int32_t* zp, const float* scale,
                               int num_channels)
      : QuantizationParams(zp, scale, num_channels) {}

  virtual int32_t get_zeroP_for_channel(int i) const override;
  virtual float get_scale_for_channel(int i) const override;
};

class alignas(alignof(uint8_t*)) QuantizationParamsHandle : public Handle {
 public:
  QuantizationParams* operator->();
  const QuantizationParams* operator->() const;
  // As long as operating on instantiations of this class and not pointers this
  // function will work
  const QuantizationParams* operator*() const;

  QuantizationParamsHandle();
  QuantizationParamsHandle(QuantizationParams* ptr);
  QuantizationParamsHandle& operator=(QuantizationParams* ptr);
  QuantizationParamsHandle(QuantizationParamsHandle&& that);
  QuantizationParamsHandle& operator=(QuantizationParamsHandle&& that);
  ~QuantizationParamsHandle();

  void free();

  //// Force everything to be on the ptensor allocator
  //void* operator new(size_t sz);
  //void operator delete(void* p);

  // KEY BIT
  friend class AllocatorInterface;
  friend class TensorInterface;
};

}  // namespace pTensor

#endif
