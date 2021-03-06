#include "errorHandler.hpp"
namespace pTensor {

Error::Error(event_handle_type t) : Event{t} {}
DEFINE_ERROR(InvalidReshapeError);
DEFINE_ERROR(InvalidResizeError);
DEFINE_ERROR(InvalidMemAccessError);
DEFINE_ERROR(OutOfMemError);
DEFINE_ERROR(OutOfMemBoundsError);
DEFINE_ERROR(InvalidOptimizableTensorError);
DEFINE_ERROR(InvalidTensorError);
DEFINE_ERROR(InvalidTensorInputError);
DEFINE_ERROR(InvalidTensorOutputError);
DEFINE_ERROR(InvalidTensorDimensionsError);
DEFINE_ERROR(InvalidTensorDataTypeError);

void ErrorHandler::uThrow(Error* err) {
  while (true) {
  }
}
void ErrorHandler::notify(const Event& evt) {
  // Do nothing
}

bool operator==(const Event& a, const Event& b) {
  return a.event_id == b.event_id;
}

}  // namespace pTensor
