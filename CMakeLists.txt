cmake_minimum_required(VERSION 3.10)

project(ptensor VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 11)
# TODO be smarter about code coverage bits
# https://github.com/codecov/example-cpp11-cmake

# pTensor never needs RTTI. Disable it for every GCC/Clang build, including
# bare-metal cross builds (which are not UNIX), since RTTI tables cost flash.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
elseif (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
endif()

#### Feature selection #########################################################
# Everything that is not needed by a target project can be left out. Operators
# are header-only templates, so an operator you never instantiate costs nothing;
# these switches cover the parts that are compiled into the library.
if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
  set(_ptensor_bare_metal_default ON)
else()
  set(_ptensor_bare_metal_default OFF)
endif()
option(PTENSOR_BARE_METAL "Target has no OS: never call exit() or use stdio" ${_ptensor_bare_metal_default})
option(PTENSOR_WITH_QUANTIZED_OPS "Build the int8 TFLM-compatible quantized operators (conv, depthwise conv, fully connected, quantize, dequantize)" ON)
option(PTENSOR_BUILD_C_API "Build the ptensor_c shared library (C API used by the Python package)" OFF)
if(PTENSOR_BUILD_C_API)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
option(PTENSOR_WITH_SIMPLE_ERROR_HANDLER "Build SimpleErrorHandler (uses std::deque and std::function)" ON)

option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -ggdb -fPIC")
endif()

option(PTENSOR_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(PTENSOR_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
endif()

option(ENABLE_PRINTING "Enable debug printing" OFF)
if(ENABLE_PRINTING)
  add_definitions(-DENABLE_PRINTING)
endif()

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

add_subdirectory(src) 
# Testing only available if this is the main app # Emergency override MODERN_CMAKE_BUILD_TESTING provided as well i
option(PACKAGE_TESTS "Build the tests" OFF)
option(PACKAGE_TUTORIALS "Build the tutorials" OFF)
#if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING) AND BUILD_TESTING)
#  #add_subdirectory(TESTS) 
#  set(PACKAGE_TESTS ON)
#endif() 
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
option(PTENSOR_SIZE_PROBE "Build a minimal app used to measure code size" OFF)
if(PTENSOR_SIZE_PROBE)
  add_subdirectory(examples/size_probe)
endif()
if(PACKAGE_TUTORIALS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  add_subdirectory(tutorials/error_handling)
  add_subdirectory(tutorials/custom_operator)
endif()

#add_subdirectory(tanh_model)