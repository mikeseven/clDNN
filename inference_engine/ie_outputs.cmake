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

option (OS_FOLDER "create OS dedicated folder in output" ON)

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (ARCH_FOLDER intel64)
else()
    set (ARCH_FOLDER  ia32)
endif()

set (BIN_FOLDER bin/${CMAKE_SYSTEM_NAME}/${ARCH_FOLDER})
set (IE_MAIN_SOURCE_DIR "${CMAKE_SOURCE_DIR}/inference_engine/dpd_vcp_dl-scoring_engine")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    debug_message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()
set (LIB_FOLDER ${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)

set (OUTPUT_ROOT ${IE_MAIN_SOURCE_DIR})