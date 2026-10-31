#file(GLOB HEADER_LIST "pTensor/allocators/*.hpp;pTensor/core/*.hpp")
# Options every pTensor library target gets: section-per-function so the
# linker's --gc-sections can drop anything the application never calls.
function(ptensor_target_defaults target)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
  endif()
  if(PTENSOR_BARE_METAL)
    target_compile_definitions(${target} PUBLIC PTENSOR_BARE_METAL=1)
  endif()
endfunction()

add_subdirectory(pTensor)
target_include_directories(ptensor 
  PUBLIC 
    $<INSTALL_INTERFACE:.>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)
#add_library(libptensor 