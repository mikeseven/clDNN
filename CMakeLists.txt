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


# Sub-directories where more CMakeLists.txt exist
add_subdirectory(tests)
add_subdirectory(src)
add_subdirectory(examples)


