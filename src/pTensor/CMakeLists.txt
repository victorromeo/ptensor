add_subdirectory(core)

option(X86_PROJECT "Build for x86" OFF)
option(ARM_PROJECT "Build for Arm" OFF)

set(src_ptensor_alloc
    allocators/arenaAllocator.cpp
   )

set(src_ptensor_tensors
  tensors/RamTensor.cpp
  tensors/BufferTensor.cpp
  tensors/RomTensor.cpp
   )

set(src_ptensor_ops
   ops/reference/MatrixMultiplyOperator.cpp
   )

set(src_ptensor_errhndl
  errorHandlers/SimpleErrorHandler.cpp
   )

#set(src_ptensor_legacy_ops
#    ops/legacy/LegacyMath.cpp
#    ops/legacy/legacyQuantizationUtils.cpp
#    )

set(src_ptensor_sq_ops
    ops/tflm/fully_connected.cpp
    ops/tflm/fully_connected_kernel.cpp
    ops/tflm/depthwise_separable_convolution.cpp
    ops/tflm/depthwise_separable_convolution_kernels.cpp
    ops/tflm/symmetric_quantization_utils.cpp
    ops/tflm/gemmlowp.cpp
    ops/tflm/convolution_helper.cpp
    ops/tflm/convolution2d.cpp
   )

if(X86_PROJECT)
add_definitions( -DX86_PROJECT=1 )
elseif(ARM_PROJECT)
add_definitions( -DARM_PROJECT=1 )
#list(APPEND src_ptensor_ops )
endif()
# add_subdirectory(ops/optimized)

add_library(ptensor  ${src_ptensor_alloc} ${src_ptensor_tensors} ${src_ptensor_ops} ${src_ptensor_errhndl} contexts/default_context.cpp ${src_ptensor_legacy_ops} ${src_ptensor_sq_ops} )
#target_link_libraries(ptensor ptensor_core optimized_ops)
target_link_libraries(ptensor PUBLIC ptensor_core )
target_include_directories(ptensor PUBLIC allocators tensors ops ops/reference ops/tflm errorHandlers)
#target_include_directories(ptensor PUBLIC 
#  $<BUILD_INTERFACE: allocators tensors ops>
#  $<INSTALL_INTERFACE: ${CMAKE_CURRENT_SOURCE_DIR}/allocators ${CMAKE_CURRENT_SOURCE_DIR}/tensors ${CMAKE_CURRENT_SOURCE_DIR}/ops>
#  )
target_compile_features(ptensor PUBLIC cxx_std_11)