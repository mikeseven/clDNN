# Copyright (c) 2016, Intel Corporation
# NeuralIA

cmake_minimum_required (VERSION 2.8)
include(CheckCXXCompilerFlag)

#Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message("=====================> ${CMAKE_BUILD_TYPE} <=====================")

set(PROJECT_NAME "NeuralIA")
project (${PROJECT_NAME})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set (COMPILER_STR "gcc")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set (COMPILER_STR "mscc")
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "CMAKE_SOURCE_DIR ...................... " ${CMAKE_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})

# To Enable C++11 gcc (Linux compiler) require additional flag to be added
if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  if(COMPILER_SUPPORTS_CXX11)
    add_definitions(-std=c++11)
  elseif(COMPILER_SUPPORTS_CXX0X)
    add_definitions(-std=c++0x)
  endif()
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC specific flags
    add_definitions(-mavx)
    add_definitions(-mavx2)
    add_definitions(-mfma)
    add_definitions(-ffast-math)
  endif()
  add_definitions(-fPIC)
else()
    add_definitions(-FIciso646) 
endif()

if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_SUPPORTS_VISIBILITY)
  if (COMPILER_SUPPORTS_VISIBILITY)
    add_definitions(-fvisibility=hidden)
  endif(COMPILER_SUPPORTS_VISIBILITY)
endif(UNIX)

# Sub-directories where more CMakeLists.txt exist
add_subdirectory(tests)
add_subdirectory(src)
add_subdirectory(examples)


