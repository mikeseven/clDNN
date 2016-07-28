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

set (TARGET_NAME "mkl_dnn")
# Sources for test files

add_definitions(-DEXPORT_NEURAL_SYMBOLS)

if (MSVC)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /arch:AVX2")
else ()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

file (GLOB API_HEADERS
       ${CMAKE_SOURCE_DIR}/api/*.h
      )
file (GLOB LIBRARY_SRC
       ${CMAKE_SOURCE_DIR}/src/*.h
       ${CMAKE_SOURCE_DIR}/src/*.cpp
      )
file (GLOB LIBRARY_SRC_GPU
       ${CMAKE_SOURCE_DIR}/src/gpu/*.h
       ${CMAKE_SOURCE_DIR}/src/gpu/*.cpp
	   ${CMAKE_SOURCE_DIR}/src/gpu/cache/*.h
       ${CMAKE_SOURCE_DIR}/src/gpu/cache/*.cpp
      )

# OpenCL includes and libraries
# Get the list of directories where libs could be stored
file(GLOB OPENCL_LIB_DIRS $ENV{INTELOCLSDKROOT}/../**/lib64/)

# Order of searching is that first SDK is checked for OpenCL headers/libs then 
# headers from project are taken and libs from hardcoded system locations
find_path(OPENCL_INCLUDE_DIRS NAMES CL/cl.hpp HINTS $ENV{INTELOCLSDKROOT}/include PATHS ../../ )
find_library(OPENCL_LIBRARIES NAMES OpenCL OpenCL.lib HINTS $ENV{INTELOCLSDKROOT}/lib/x64 $ENV{INTELOCLSDKROOT}
 ${OPENCL_LIB_DIRS} PATHS /lib64 /usr/lib )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("api" FILES ${API_HEADERS})
source_group("src" FILES ${LIBRARY_SRC})
source_group("src\\gpu" FILES ${LIBRARY_SRC_GPU})

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_SOURCE_DIR}
                     ${CMAKE_SOURCE_DIR}/src
                     ${OPENCL_INCLUDE_DIRS}
                    )

# Create library file from sources.
add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${LIBRARY_SRC_GPU}
            ${API_HEADERS})

target_link_libraries(${TARGET_NAME} ${OPENCL_LIBRARIES} )

if(UNIX)
    target_link_libraries(${TARGET_NAME} pthread)
endif()

