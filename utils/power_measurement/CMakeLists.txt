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

set(CLDNN_BUILD__PROJ             "power_measurement_util")
set(CLDNN_BUILD__PROJ_LABEL       "power measurement")
set(CLDNN_BUILD__PROJ_FOLDER      "utils")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "pmeasure_util${CLDNN__OUT_CPU_SUFFIX}")

# Published to parent scope.
set(CLDNN_UTILS__POWER_MEASUREMENT_INCDIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CLDNN_UTILS__POWER_MEASUREMENT_LIBDIRS "")

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      RttiEnabled
      WarnLevel3
      TreatWarnAsErrorDisabled
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main        "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__include "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(__CLDNN_Label__include     "include")
file(GLOB __CLDNN_Sources__include
    "${__CLDNN_Directory__include}/*.h"
    "${__CLDNN_Directory__include}/*.hpp"
  )

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__include}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"    FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__include}" FILES ${__CLDNN_Sources__include})

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN_UTILS__POWER_MEASUREMENT_INCDIRS}
  )

# =================================== Link targets and dependencies ====================================

# Power measurement utility library (static).
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY FOLDER        "${CLDNN_BUILD__PROJ_FOLDER}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})

# ======================================================================================================

# Publishing some information to parent scope.
set(CLDNN_UTILS__POWER_MEASUREMENT_INCDIRS ${CLDNN_UTILS__POWER_MEASUREMENT_INCDIRS} PARENT_SCOPE)
set(CLDNN_UTILS__POWER_MEASUREMENT_LIBDIRS ${CLDNN_UTILS__POWER_MEASUREMENT_LIBDIRS} PARENT_SCOPE)

# ======================================================================================================
