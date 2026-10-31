#include "quantizationPrimitives.hpp"
#include "context.hpp"
#include "memoryManagementInterface.hpp"
#include "pTensor_util.hpp"

namespace pTensor {

DEFINE_ERROR(AttemptToUseUnSpecializedQuantizeParamsError);

QuantizationParams::QuantizationParams()
    : _zp(nullptr), _scale(nullptr), _num_channels(0) {}
QuantizationParams::QuantizationParams(const int32_t* zp, const float* scale,
                                       int num_channels)
    : _zp(zp), _scale(scale), _num_channels(num_channels) {}

QuantizationParams::~QuantizationParams() {}
int32_t QuantizationParams::get_zeroP_for_channel(int i) const { 
  pTensor_printf(
      "ERROR, Attempted to use non specialized Quantization params\n");
  Context::get_default_context()->throwError(AttemptToUseUnSpecializedQuantizeParamsError());
  return 0; 
}
float QuantizationParams::get_scale_for_channel(int i) const { 
  pTensor_printf(
      "ERROR, Attempted to use non specialized Quantization params\n");
  Context::get_default_context()->throwError(AttemptToUseUnSpecializedQuantizeParamsError());
  return 0; 
}
// Allocate the tensor metadata on a different heap from the data scratch pads
void* QuantizationParams::operator new(size_t sz) {
  void* p =
      Context::get_default_context()->get_metadata_allocator()->allocate(sz);
  return p;
}

void QuantizationParams::operator delete(void* p) {
  Context::get_default_context()->get_metadata_allocator()->deallocate(p);
}

PerTensorQuantizationParams::PerTensorQuantizationParams(int32_t zp,
                                                         float scale)
    : QuantizationParams(&_zp_value, &_scale_value, 1),
      _zp_value(zp),
      _scale_value(scale) {}
PerTensorQuantizationParams::PerTensorQuantizationParams(
    const PerTensorQuantizationParams& that)
    : QuantizationParams(&_zp_value, &_scale_value, 1),
      _zp_value(that._zp_value),
      _scale_value(that._scale_value) {}
PerTensorQuantizationParams& PerTensorQuantizationParams::operator=(
    const PerTensorQuantizationParams& that) {
  _zp_value = that._zp_value;
  _scale_value = that._scale_value;
  return *this;
}
int32_t PerTensorQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp_value;
}
float PerTensorQuantizationParams::get_scale_for_channel(int i) const {
  return _scale_value;
}

int32_t PerChannelQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp[i];
}
float PerChannelQuantizationParams::get_scale_for_channel(int i) const {
  return _scale[i];
}

QuantizationParams* QuantizationParamsHandle::operator->() {
  return reinterpret_cast<QuantizationParams*>(_ptr);
}
const QuantizationParams* QuantizationParamsHandle::operator->() const {
  return reinterpret_cast<const QuantizationParams*>(_ptr);
}
const QuantizationParams* QuantizationParamsHandle::operator*() const {
  return reinterpret_cast<QuantizationParams*>(_ptr);
}
QuantizationParamsHandle::~QuantizationParamsHandle() { free(); }
void QuantizationParamsHandle::free() {
  void* ptr_t = _ptr; //unbind invalidates this handle so store a copy
  if (_ptr) {
    AllocatorInterface* alloc =
        Context::get_default_context()->get_metadata_allocator();
    if (alloc->is_bound(_ptr, this)) {
      alloc->unbind(_ptr, this);

    }
    delete reinterpret_cast<QuantizationParams*>(ptr_t);
    alloc->deallocate(ptr_t);
  }
  _ptr = nullptr;
}
QuantizationParamsHandle::QuantizationParamsHandle() : Handle() {}
QuantizationParamsHandle::QuantizationParamsHandle(QuantizationParams* ptr) : Handle((void*)ptr) {
  // Context::get_default_context()->get_metadata_allocator()->bind(_ptr, this);
  bind(*this, *Context::get_default_context()->get_metadata_allocator());
}
QuantizationParamsHandle& QuantizationParamsHandle::operator=(QuantizationParams* ptr) {
  _ptr = (void*)ptr;
  bind(*this, *Context::get_default_context()->get_metadata_allocator());
  // Context::get_metadata_allocator()->bind(_ptr, this);
  return *this;
}

QuantizationParamsHandle::QuantizationParamsHandle(QuantizationParamsHandle&& that) {
  _ptr = that._ptr;
  AllocatorInterface* alloc =
      Context::get_default_context()->get_metadata_allocator();
  if (alloc->is_bound(_ptr, &that)) {
    alloc->unbind(_ptr, &that);
    alloc->bind(_ptr, this);
  }
  that._ptr = nullptr;
}
QuantizationParamsHandle& QuantizationParamsHandle::operator=(QuantizationParamsHandle&& that) {
  if (this != &that) {
    _ptr = that._ptr;
    AllocatorInterface* alloc =
        Context::get_default_context()->get_metadata_allocator();
    if (alloc->is_bound(_ptr, &that)) {
      alloc->unbind(_ptr, &that);
      alloc->bind(_ptr, this);
    }
    that._ptr = nullptr;
  }
  return *this;
}
// Add some bits to make the interface nicer to the user


}  // namespace pTensor
