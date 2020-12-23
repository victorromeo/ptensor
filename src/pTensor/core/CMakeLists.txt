set(src_ptensor_core
    types.cpp
    pTensor_util.cpp
    tensorBase.cpp
    memoryManagementInterface.cpp
    context.cpp
    tensor.cpp
    TensorMap.cpp
    operatorBase.cpp
    errorHandler.cpp
    modelBase.cpp
    quantizationPrimitives.cpp
   )

set(hdr_ptensor_core
    types.hpp
    pTensor_util.hpp
    tensorBase.hpp
    memoryManagementInterface.hpp
    context.hpp
    tensor.hpp
    TensorMap.hpp
    operatorBase.hpp
    ptensor_string.hpp
    errorHandler.hpp
    modelBase.hpp
    quantizationPrimitives.hpp
   )
    
add_library(ptensor_core ${src_ptensor_core} )
target_include_directories(ptensor_core PUBLIC .)
target_compile_features(ptensor_core PUBLIC cxx_std_11)
