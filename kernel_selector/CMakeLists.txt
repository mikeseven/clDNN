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

if(CLDNN_BUILT_FROM_OPENVX)
  FIND_PACKAGE(OpenCL)
  if (NOT OpenCL_FOUND)
    message(STATUS "Couldn't build clDNN library. Skip plugin build." )
    return( )
  endif( )
endif( )

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "cldnn_kernel_selector")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# ================================== Compiler preprocessor definitions =================================

if (CLDNN__ENABLE_PERSISTENT_CACHE)
  set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS CLDNN_ENABLE_PERSISTENT_CACHE)
endif ()

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__api                 "api")
file(GLOB __CLDNN_Headers__api
    "${CLDNN__API_DIR}/*.h"
    "${CLDNN__API_DIR}/*.hpp"
  )

set(__CLDNN_Directory__api__cpp "${CLDNN__API_DIR}/CPP")
set(__CLDNN_Label__api__primitives     "${__CLDNN_Label__api}\\CPP")
file(GLOB __CLDNN_Headers__api__cpp
    "${__CLDNN_Directory__api__cpp}/*.h"
    "${__CLDNN_Directory__api__cpp}/*.hpp"
  )

set(__CLDNN_Directory__api__c "${CLDNN__API_DIR}/C")
set(__CLDNN_Label__api__c     "${__CLDNN_Label__api}\\C")
file(GLOB __CLDNN_Headers__api__c
    "${__CLDNN_Directory__api__c}/*.h"
    "${__CLDNN_Directory__api__c}/*.hpp"
  )

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )
  
set(__CLDNN_Directory__core           "${CMAKE_CURRENT_SOURCE_DIR}/core")
set(__CLDNN_Label__core               "core")
file(GLOB __CLDNN_Sources__core
    "${__CLDNN_Directory__core}/*.h"
    "${__CLDNN_Directory__core}/*.hpp"
    "${__CLDNN_Directory__core}/*.cpp"
  )

set(__CLDNN_Directory__common          "${CMAKE_CURRENT_SOURCE_DIR}/common")
set(__CLDNN_Label__common              "common")
file(GLOB __CLDNN_Sources__common
    "${__CLDNN_Directory__common}/*.h"
    "${__CLDNN_Directory__common}/*.hpp"
    "${__CLDNN_Directory__common}/*.cpp"
  )
  
set(__CLDNN_Directory__core_common          "${__CLDNN_Directory__core}/common")
set(__CLDNN_Label__core_common              "${__CLDNN_Label__core}\\common")
file(GLOB __CLDNN_Sources__core_common
    "${__CLDNN_Directory__core_common}/*.h"
    "${__CLDNN_Directory__core_common}/*.hpp"
    "${__CLDNN_Directory__core_common}/*.cpp"
  )

set(__CLDNN_Directory__actual_kernels "${__CLDNN_Directory__core}/actual_kernels")
set(__CLDNN_Label__actual_kernels     "${__CLDNN_Label__core}\\actual_kernels")
file(GLOB_RECURSE __CLDNN_Sources__actual_kernels
    "${__CLDNN_Directory__actual_kernels}/*.h"
    "${__CLDNN_Directory__actual_kernels}/*.hpp"
    "${__CLDNN_Directory__actual_kernels}/*.cpp"
  )
  
set(__CLDNN_Directory__vx_adapter   "${CMAKE_CURRENT_SOURCE_DIR}/vx_adapter")
set(__CLDNN_Label__vx_adapter       "vx_adapter")
file(GLOB __CLDNN_Sources__vx_adapter 
    "${__CLDNN_Directory__vx_adapter}/api/*.h"
    "${__CLDNN_Directory__vx_adapter}/*.h"
    "${__CLDNN_Directory__vx_adapter}/*.hpp"
    "${__CLDNN_Directory__vx_adapter}/*.cpp"
  )

set(__CLDNN_Directory__cache           "${__CLDNN_Directory__vx_adapter}/cache")
set(__CLDNN_Label__cache               "${__CLDNN_Label__vx_adapter}\\cache")
file(GLOB __CLDNN_Sources__cache
    "${__CLDNN_Directory__cache}/*.h"
    "${__CLDNN_Directory__cache}/*.hpp"
    "${__CLDNN_Directory__cache}/*.cpp"
  )

set(__CLDNN_Directory__ch_kernels      "${__CLDNN_Directory__core}/cl_kernels")
set(__CLDNN_Label__ch_kernels          "${__CLDNN_Label__core}\\cl_kernels")
file(GLOB_RECURSE __CLDNN_Sources__ch_kernels
    "${__CLDNN_Directory__ch_kernels}/*.cl"
  )

set(__CLDNN_Directory__cg_cache        "${CLDNN__CODEGEN_INCDIR}")
set(__CLDNN_CGDirectory__cg_cache      "${CLDNN__CODEGEN_DIR}/cache")
set(__CLDNN_Label__cg_cache            "${__CLDNN_Label__cache}\\codegen")
set(__CLDNN_File__cg_cache__prim_db    "ks_primitive_db.inc")
set(__CLDNN_Sources__cg_cache
    "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Headers__api}
    ${__CLDNN_Headers__api__cpp}
    ${__CLDNN_Headers__api__c}
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__core}
    ${__CLDNN_Sources__common}
    ${__CLDNN_Sources__core_common}
    ${__CLDNN_Sources__actual_kernels}
    ${__CLDNN_Sources__cache}
    ${__CLDNN_Sources__ch_kernels}
    ${__CLDNN_Sources__cg_cache}
    ${__CLDNN_Sources__vx_adapter}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__api}"             FILES ${__CLDNN_Headers__api})
source_group("${__CLDNN_Label__api__cpp}" FILES ${__CLDNN_Headers__api__cpp})
source_group("${__CLDNN_Label__api__c}" FILES ${__CLDNN_Headers__api__c})
source_group("${__CLDNN_Label__main}"            FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__core}"            FILES ${__CLDNN_Sources__core})
source_group("${__CLDNN_Label__common}"          FILES ${__CLDNN_Sources__common})
source_group("${__CLDNN_Label__core_common}"     FILES ${__CLDNN_Sources__core_common})
source_group("${__CLDNN_Label__actual_kernels}"  FILES ${__CLDNN_Sources__actual_kernels})
source_group("${__CLDNN_Label__cache}"           FILES ${__CLDNN_Sources__cache})
source_group("${__CLDNN_Label__ch_kernels}"      FILES ${__CLDNN_Sources__ch_kernels})
source_group("${__CLDNN_Label__cg_cache}"        FILES ${__CLDNN_Sources__cg_cache})
source_group("${__CLDNN_Label__vx_adapter}"   FILES ${__CLDNN_Sources__vx_adapter})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${__CLDNN_Directory__core}"
    "${__CLDNN_Directory__core}/common"
    "${__CLDNN_Directory__actual_kernels}"
    "${__CLDNN_Directory__common}"
  )

# =================================== Link targets and dependencies ====================================

# Main shared library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")


if(CLDNN_BUILT_FROM_OPENVX)
    target_link_libraries("${CLDNN_BUILD__PROJ}" 
        ${OpenCL_LIBRARIES} 
      )
else()
    target_link_libraries("${CLDNN_BUILD__PROJ}"
        OpenCL
      )
endif()

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif(UNIX)
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()

# =================================== Custom pre- and post-steps =======================================

add_custom_command(OUTPUT "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${__CLDNN_CGDirectory__cg_cache}"
    COMMAND "${PYTHON_EXECUTABLE}" "${__CLDNN_Directory__core_common}/primitive_db_gen.py" -out_path "${__CLDNN_CGDirectory__cg_cache}" -out_file_name "${__CLDNN_File__cg_cache__prim_db}" -kernels "${__CLDNN_Directory__ch_kernels}"
    DEPENDS ${__CLDNN_Sources__ch_kernels} "${__CLDNN_Directory__core_common}/primitive_db_gen.py"
    COMMENT "Generating ${__CLDNN_File__cg_cache__prim_db} ..."
  )
add_custom_command(OUTPUT "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    DEPENDS "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" ${__CLDNN_Sources__ch_kernels} "${__CLDNN_Directory__core_common}/primitive_db_gen.py"
    COMMENT "Updating file if the file changed (${__CLDNN_File__cg_cache__prim_db}) ..."
  )

# ======================================================================================================
