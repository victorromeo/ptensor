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
    
# Generated header recording which optional features this build includes.
set(PTENSOR_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(ptensor_config.h.in "${PTENSOR_CONFIG_DIR}/pTensor/ptensor_config.h")

add_library(ptensor_core ${src_ptensor_core} )
ptensor_target_defaults(ptensor_core)
target_include_directories(ptensor_core PUBLIC
  .
  $<BUILD_INTERFACE:${PTENSOR_CONFIG_DIR}>
  $<INSTALL_INTERFACE:include>)
target_compile_features(ptensor_core PUBLIC cxx_std_11)

install(FILES "${PTENSOR_CONFIG_DIR}/pTensor/ptensor_config.h"
        DESTINATION include/pTensor/pTensor)
