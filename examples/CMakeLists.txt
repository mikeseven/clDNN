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

set(CLDNN_BUILD__PROJ             "examples")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      RttiEnabled
      WarnLevel3
      TreatWarnAsErrorDisabled
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__common          "${CMAKE_CURRENT_SOURCE_DIR}/common")
set(__CLDNN_Label__common              "common")
file(GLOB __CLDNN_Sources__common
    "${__CLDNN_Directory__common}/*.h"
    "${__CLDNN_Directory__common}/*.hpp"
    "${__CLDNN_Directory__common}/*.cpp"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__common}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"   FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__common}" FILES ${__CLDNN_Sources__common})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

# =================================== Link targets and dependencies ====================================

# Examples executable.
add_executable("${CLDNN_BUILD__PROJ}"
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

target_link_libraries("${CLDNN_BUILD__PROJ}"
    "${CLDNN_BUILD__PROJ__clDNN}"
    FreeImage
    Boost::filesystem
    Boost::program_options
    Boost::system
  )

# =================================== Custom pre- and post-steps =======================================

set(__CLDNN_FindLibSuffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
find_library(CLDNN_FIND__FREEIMAGE_SHLPATH
  NAMES FreeImage
  PATHS ${CLDNN__FREEIMAGE_LIBDIRS}
  NO_CMAKE_FIND_ROOT_PATH
)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${__CLDNN_FindLibSuffixes})

if(CLDNN_FIND__FREEIMAGE_SHLPATH)
  add_custom_command(TARGET "${CLDNN_BUILD__PROJ}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CLDNN_FIND__FREEIMAGE_SHLPATH}" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/"
      COMMENT "Copying FreeImage shared library to output..."
    )
endif()

unset(CLDNN_FIND__FREEIMAGE_SHLPATH CACHE)
unset(__CLDNN_FindLibSuffixes)

# ======================================================================================================
