#ifndef PTENSOR_FULLY_CONNECTED_H
#define PTENSOR_FULLY_CONNECTED_H

#include "MatrixMultiplyOperator.hpp"

namespace pTensor {

namespace ReferenceOperators {

template <typename Tout>
using FullyConnectedOperator = MatrixMultOperatorV2<Tout>;

}

}

#endif