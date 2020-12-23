#ifndef PTENSOR_SIMPLE_ERROR_HANDLER_HPP
#define PTENSOR_SIMPLE_ERROR_HANDLER_HPP
#include <deque>
#include <functional>

#include "errorHandler.hpp"

namespace pTensor {

class SimpleErrorHandler : public ErrorHandler {
 public:
  using onErrorF_t = std::function<void(Error*)>;

  SimpleErrorHandler(size_t max_num_events);
  virtual void uThrow(Error* err) override;
  virtual void notify(const Event& evt) override;
  void set_onError(onErrorF_t onError);
  std::deque<Event>::iterator begin();
  std::deque<Event>::iterator end();

 private:
  std::deque<Event> _eventQ;
  onErrorF_t _onError;
  size_t max_num_events;
};

}  // namespace pTensor
#endif
