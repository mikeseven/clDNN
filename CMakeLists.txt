# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 2.8)
include(CheckCXXCompilerFlag)

#Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message("=====================> ${CMAKE_BUILD_TYPE} <=====================")

set(PROJECT_NAME "MKL-DNN")
project (${PROJECT_NAME})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set (COMPILER_STR "gcc")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set (COMPILER_STR "mscc")
endif()
if(NOT(UNIX))
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
  set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
  set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY})
endif()

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
  elseif ((NOT(COMPILER_SUPPORTS_CXX11)) AND (NOT(COMPILER_SUPPORTS_CXX0X)))
    message (FATAL_ERROR "CMake BREAK: Compiler must support C++11")
  endif()
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") # GCC specific flags
    add_definitions(-mavx)
    add_definitions(-mavx2)
    add_definitions(-mfma)
    add_definitions(-ffast-math)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      add_definitions(-fstack-protector-strong)
    else()
      add_definitions(-fstack-protector)
    endif()
  endif()
  add_definitions(-fno-operator-names)
  add_definitions(-z noexecstack)
  add_definitions(-z relro)
  add_definitions(-z now)
  add_definitions(-Wformat)
  add_definitions(-Wformat-security)
  if( CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-D_FORTIFY_SOURCE=2)
  endif()
else()
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC") #MSVC specific flags
    add_definitions(/sdl)
  endif()
endif()

if(UNIX)
  CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_SUPPORTS_VISIBILITY)
  if (COMPILER_SUPPORTS_VISIBILITY)
    #add_definitions(-fvisibility=hidden) todo: should be hidden? if so define default visibiliti explicite for each funtion
    add_definitions(-fvisibility=default) 
    endif(COMPILER_SUPPORTS_VISIBILITY)
endif(UNIX)

include_directories(${CMAKE_SOURCE_DIR}/api)
include_directories(${CMAKE_SOURCE_DIR}/xbyak)

# Sub-directories where more CMakeLists.txt exist
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
