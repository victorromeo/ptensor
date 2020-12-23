#file(GLOB HEADER_LIST "pTensor/allocators/*.hpp;pTensor/core/*.hpp")
add_subdirectory(pTensor)
target_include_directories(ptensor 
  PUBLIC 
    $<INSTALL_INTERFACE:.>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)
#add_library(libptensor 