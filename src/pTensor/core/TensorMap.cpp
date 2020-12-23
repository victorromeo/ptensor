#include "TensorMap.hpp"

namespace pTensor {
const pTensor::string not_found_string("NotFound");
Tensor not_found_tensor(nullptr);

SimpleNamedTensor TensorMapInterface::not_found(not_found_string,
                                                not_found_tensor);

bool compare_named_tensors(const SimpleNamedTensor& a,
                           const SimpleNamedTensor& b) {
  return a.name < b.name;
}

}  // namespace pTensor
