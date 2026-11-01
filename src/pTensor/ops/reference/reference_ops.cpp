#include "ElementwiseOperators.hpp"
#include "LayerOperators.hpp"
#include "ReLuOperator.hpp"

namespace pTensor {
DEFINE_ERROR(BroadcastShapeMismatchError);
DEFINE_ERROR(QuantizedActivationScaleError);
DEFINE_ERROR(InvalidLayerShapeError);
}  // namespace pTensor
