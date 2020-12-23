#ifndef __pTensor_H
#define __pTensor_H
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

/*
 * Operators
 */

#include "pTensor/ops/tflm/convolution2d.hpp"
#include "pTensor/ops/tflm/depthwise_separable_convolution.hpp"
#include "pTensor/ops/tflm/fully_connected.hpp"
#include "pTensor/ops/tflm/DequantizeOperator.hpp"
#include "pTensor/ops/tflm/QuantizeOperator.hpp"

// #include "pTensor/ops/tanh.hpp"

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


/*
 * Tensors
 */

#include "pTensor/tensors/BufferTensor.hpp"
#include "pTensor/tensors/RamTensor.hpp"
#include "pTensor/tensors/RomTensor.hpp"

/*
 * Error Handlers
 */
#include "pTensor/errorHandlers/SimpleErrorHandler.hpp"

#endif

