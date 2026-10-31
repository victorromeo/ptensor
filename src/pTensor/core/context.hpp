#ifndef PTENSOR_CTX_H
#define PTENSOR_CTX_H
#include "errorHandler.hpp"
#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"
#include "pTensor_util.hpp"

namespace pTensor {

DECLARE_ERROR(MemoryAllocatorUnsetError);

class Context {
 private:
  Context();

 public:
  static Context* get_default_context();
  ErrorHandler* get_default_error_handler();
  AllocatorInterface* get_metadata_allocator();
  AllocatorInterface* get_ram_data_allocator();
  void set_metadata_allocator(AllocatorInterface* al);
  void set_ram_data_allocator(AllocatorInterface* al);
  void set_ErrorHandler(ErrorHandler* errH);
  void register_tensor(TensorBase* tb);
  // Report an error to the active handler. Errors are passed by reference so
  // that raising one never touches the heap (important when the error is
  // itself an out-of-memory condition). The Error is only valid for the
  // duration of the handler call.
  void throwError(const Error& err);
  // Legacy form: takes ownership of a heap-allocated Error and deletes it once
  // the handler returns.
  void throwError(Error* err);
  void notifyEvent(const Event& err);

 private:
  AllocatorInterface* _metadata_allocator;
  AllocatorInterface* _ram_data_allocator;
  ErrorHandler* _error_handler;
};
}  // namespace pTensor
#endif  // PTENSOR_CTX_H
