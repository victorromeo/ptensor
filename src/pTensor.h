#ifndef __pTensor_H
#define __pTensor_H
/*
 * Build configuration (which optional features were compiled in)
 */
#include "pTensor/ptensor_config.h"

/*
 * Core bits
 */
// This one selects platform specific stuff. probably should come first
#include "pTensor/core/modelBase.hpp"
#include "pTensor/core/operatorBase.hpp"
#include "pTensor/core/pTensor_util.hpp"

/*
 * Allocators
 */
#include "pTensor/allocators/arenaAllocator.hpp"
#include "pTensor/allocators/arenaPlanner.hpp"

/*
 * Operators
 */

#if PTENSOR_WITH_QUANTIZED_OPS
#include "pTensor/ops/tflm/convolution2d.hpp"
#include "pTensor/ops/tflm/depthwise_separable_convolution.hpp"
#include "pTensor/ops/tflm/fully_connected.hpp"
#include "pTensor/ops/tflm/DequantizeOperator.hpp"
#include "pTensor/ops/tflm/QuantizeOperator.hpp"
#endif

#include "pTensor/ops/reference/TanhOperator.hpp"

#include "pTensor/ops/reference/ReLuOperator.hpp"
#include "pTensor/ops/reference/SoftMaxOperator.hpp"
#include "pTensor/ops/reference/SigmoidOperator.hpp"
#include "pTensor/ops/reference/LogisticOperator.hpp"
#include "pTensor/ops/reference/AddOperator.hpp"
#include "pTensor/ops/reference/ArgMaxOperator.hpp"
#include "pTensor/ops/reference/ArgMinOperator.hpp"
#include "pTensor/ops/reference/MatrixMultiplyOperator.hpp"
#include "pTensor/ops/reference/MaxOperator.hpp"
#include "pTensor/ops/reference/MinOperator.hpp"
#include "pTensor/ops/reference/SqueezeOperator.hpp"
#include "pTensor/ops/reference/ConvolutionOperator.hpp"
#include "pTensor/ops/reference/FullyConnectedOperator.hpp"
#include "pTensor/ops/reference/AvgPoolOperator.hpp"
#include "pTensor/ops/reference/MaxPoolOperator.hpp"
#include "pTensor/ops/reference/MinPoolOperator.hpp"
#include "pTensor/ops/reference/TransposeOperator.hpp"
#include "pTensor/ops/reference/ReshapeOperator.hpp"
#include "pTensor/ops/reference/ElementwiseOperators.hpp"
#include "pTensor/ops/reference/LayerOperators.hpp"

#if PTENSOR_WITH_FUSION
#include "pTensor/ops/fused/FusedOperators.hpp"
#include "pTensor/util/fusion.hpp"
#if PTENSOR_WITH_QUANTIZED_OPS
#include "pTensor/ops/fused/QuantizedAddOperator.hpp"
#endif
#endif


/*
 * Tensors
 */

#include "pTensor/tensors/BufferTensor.hpp"
#include "pTensor/tensors/RamTensor.hpp"
#include "pTensor/tensors/RomTensor.hpp"
#include "pTensor/tensors/PackedRomTensor.hpp"
#include "pTensor/util/packing.hpp"
#if PTENSOR_WITH_POLY_QUANT
#include "pTensor/tensors/PolyRomTensor.hpp"
#include "pTensor/util/polyquant.hpp"
#endif
#if PTENSOR_WITH_HOUGH
#include "pTensor/tensors/HoughRomTensor.hpp"
#include "pTensor/util/hough.hpp"
#endif

/*
 * Error Handlers
 */
#if PTENSOR_WITH_SIMPLE_ERROR_HANDLER
#include "pTensor/errorHandlers/SimpleErrorHandler.hpp"
#endif

#endif

