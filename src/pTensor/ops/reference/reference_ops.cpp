#include "ElementwiseOperators.hpp"
#include "LayerOperators.hpp"
#include "ReLuOperator.hpp"
#include "SliceOperator.hpp"

namespace pTensor {
DEFINE_ERROR(BroadcastShapeMismatchError);
DEFINE_ERROR(QuantizedActivationScaleError);
DEFINE_ERROR(InvalidLayerShapeError);
DEFINE_ERROR(InvalidSliceError);
}  // namespace pTensor
