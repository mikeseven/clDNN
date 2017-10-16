# Copyright (c) 2017 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 2.8)
include(../ie_outputs.cmake)
include(../ie_imports.cmake)
set (TARGET_NAME "generic_sample")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fPIE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fPIE")

if (ENABLE_SAMPLES)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
		${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common/power_instrumentation.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common/power_instrumentation.h
        )

if (UNIX)
    SET(LIB_DL dl)
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../format_reader
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include)

link_directories(${CMAKE_SOURCE_DIR}/${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

add_dependencies(${TARGET_NAME} gflags clDNNPlugin)


target_link_libraries(${TARGET_NAME} format_reader inference_engine gflags)

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
else()
endif()
endif(ENABLE_SAMPLES)
