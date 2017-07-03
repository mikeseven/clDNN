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

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "cost_model")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      WarnLevel3
      NoFastMath
  )
if (NOT MSVC)
  intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
      SET_RAW
        "-Wno-error=conversion-null"
        "-Wno-error=type-limits"
    )
endif ()

# ========================================= Source/Header files ========================================

set(CLDNN__KERNEL_SELECTOR_DIR      "${CLDNN__MAIN_DIR}/kernel_selector")

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )
  
set(__CLDNN_Directory__core           "${CLDNN__KERNEL_SELECTOR_DIR}/core")
set(__CLDNN_Label__core               "core")
file(GLOB_RECURSE __CLDNN_Sources__core
    "${__CLDNN_Directory__core}/*.h"
    "${__CLDNN_Directory__core}/*.hpp"
  )

set(__CLDNN_Directory__common          "${CLDNN__KERNEL_SELECTOR_DIR}/common")
set(__CLDNN_Label__common              "common")
file(GLOB __CLDNN_Sources__common
    "${__CLDNN_Directory__common}/*.h"
    "${__CLDNN_Directory__common}/*.hpp"
  )

set(__CLDNN_Directory__cache           "${CLDNN__KERNEL_SELECTOR_DIR}/vx_adapter/cache")
set(__CLDNN_Label__cache               "cache")
file(GLOB __CLDNN_Sources__cache
    "${__CLDNN_Directory__cache}/*.h"
    "${__CLDNN_Directory__cache}/*.hpp"
  )

set(__CLDNN_Directory__ch_kernels      "${CLDNN__KERNEL_SELECTOR_DIR}/cl_kernels")
set(__CLDNN_Label__ch_kernels          "cl_kernels")
file(GLOB_RECURSE __CLDNN_Sources__ch_kernels
    "${__CLDNN_Directory__ch_kernels}/*.cl"
  )
  
set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__core}
    ${__CLDNN_Sources__cache}
    ${__CLDNN_Sources__common}
    ${__CLDNN_Sources__ch_kernels}
  )
# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"            FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__cache}"           FILES ${__CLDNN_Sources__cache})
source_group("${__CLDNN_Label__gpu}"             FILES ${__CLDNN_Sources__gpu})
source_group("${__CLDNN_Label__ch_kernels}"      FILES ${__CLDNN_Sources__ch_kernels})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLDNN__KERNEL_SELECTOR_DIR}"
    "${__CLDNN_Directory__core}"
    "${__CLDNN_Directory__core}/common"
    "${__CLDNN_Directory__common}"
    "${__CLDNN_Directory__cache}"
  )

# =================================== Link targets and dependencies ====================================

# Tests executable.
add_executable("${CLDNN_BUILD__PROJ}"
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")


# Set library dependencies
target_link_libraries("${CLDNN_BUILD__PROJ}"
    OpenCL
    cldnn_kernel_selector
  )
  
add_dependencies("${CLDNN_BUILD__PROJ}"
    cldnn_kernel_selector
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif(UNIX)
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()

