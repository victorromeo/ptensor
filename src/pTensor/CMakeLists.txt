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
  tensors/PackedRomTensor.cpp
   )

set(src_ptensor_ops
   ops/reference/MatrixMultiplyOperator.cpp
   ops/reference/reference_ops.cpp
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

if(NOT PTENSOR_WITH_QUANTIZED_OPS)
  set(src_ptensor_sq_ops)
endif()
if(NOT PTENSOR_WITH_SIMPLE_ERROR_HANDLER)
  set(src_ptensor_errhndl)
endif()

add_library(ptensor  ${src_ptensor_alloc} ${src_ptensor_tensors} ${src_ptensor_ops} ${src_ptensor_errhndl} contexts/default_context.cpp ${src_ptensor_legacy_ops} ${src_ptensor_sq_ops} )
ptensor_target_defaults(ptensor)
#target_link_libraries(ptensor ptensor_core optimized_ops)
target_link_libraries(ptensor PUBLIC ptensor_core )
target_include_directories(ptensor PUBLIC allocators tensors ops ops/reference ops/tflm errorHandlers)
target_compile_features(ptensor PUBLIC cxx_std_11)

if(PTENSOR_BUILD_C_API)
  add_library(ptensor_c SHARED c_api/ptensor_c.cpp)
  target_link_libraries(ptensor_c PRIVATE ptensor)
  set_target_properties(ptensor_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  install(TARGETS ptensor_c LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
endif()

install(TARGETS ptensor ptensor_core
  EXPORT ptensorTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

INSTALL (
   DIRECTORY ${CMAKE_SOURCE_DIR}/src/
   DESTINATION include/pTensor
   FILES_MATCHING PATTERN "*.h*")