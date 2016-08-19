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


add_subdirectory(gtest)

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "tests")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET WarnLevel3
  )
if (NOT MSVC)
  intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
      SET_RAW
        "-Wno-error=conversion-null"
        "-Wno-error=type-limits"
    )
endif ()

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__test_cases      "${CMAKE_CURRENT_SOURCE_DIR}/test_cases")
set(__CLDNN_Label__test_cases          "test cases")
file(GLOB __CLDNN_Sources__test_cases
    "${__CLDNN_Directory__test_cases}/*.h"
    "${__CLDNN_Directory__test_cases}/*.hpp"
    "${__CLDNN_Directory__test_cases}/*.cpp"
  )

set(__CLDNN_Directory__test_utils      "${CMAKE_CURRENT_SOURCE_DIR}/test_utils")
set(__CLDNN_Label__test_utils          "test utils")
file(GLOB __CLDNN_Sources__test_utils
    "${__CLDNN_Directory__test_utils}/*.h"
    "${__CLDNN_Directory__test_utils}/*.hpp"
    "${__CLDNN_Directory__test_utils}/*.cpp"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__test_cases}
    ${__CLDNN_Sources__test_utils}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"       FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__test_cases}" FILES ${__CLDNN_Sources__test_cases})
source_group("${__CLDNN_Label__test_utils}" FILES ${__CLDNN_Sources__test_utils})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CLDNN__MAIN_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${__CLDNN_Directory__test_utils}"
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
    gtest
    "${CLDNN_BUILD__PROJ__clDNN}"
  )

# ======================================================================================================

# TODO: Clean up unused / unnecessary custom steps.


# To use SDE it has to be installed and its root dir needs to be
# added to PATH env variable.Also USE_SDE_EMULATION should be set
# to ON (see scripts for generating build system)
if(USE_SDE_EMULATION)
find_program(SDE_EXE_DIR ${SDE_BINARY_NAME})
message("SDE: ${SDE_EXE_DIR}")
endif()

if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
set(TEST_COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CLDNN_BUILD__PROJ})
else()
set(TEST_COMMAND ${SDE_EXE_DIR} -bdw -- ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CLDNN_BUILD__PROJ})
endif()

set(GFLAGS_BINARY_NAME gflags)

if(RUN_ULTS_OFFLINE)
    add_test(${CLDNN_BUILD__PROJ} ${TEST_COMMAND})
    message (STATUS "add_test ...................... " ${CLDNN_BUILD__PROJ} ${TEST_COMMAND})
elseif(CLDNN__RUN_TESTS)
    if(NOT UNIX)
        find_program(GFLAGS_EXE_DIR ${GFLAGS_BINARY_NAME})
        message("GFLAGS: ${GFLAGS_EXE_DIR}")

        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${CLDNN_BUILD__PROJ}
                           POST_BUILD
                           WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${CLDNN_BUILD__PROJ}.exe +ust +hpa )
        endif()
    endif()

    if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
        add_custom_command(TARGET ${CLDNN_BUILD__PROJ}
                       POST_BUILD
                       WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                       COMMAND ${CLDNN_BUILD__PROJ} )
    else()
        add_custom_command(TARGET ${CLDNN_BUILD__PROJ}
                       POST_BUILD
                       WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                       COMMAND ${CLDNN_BUILD__PROJ} )
    endif()

    if(NOT UNIX)
        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${CLDNN_BUILD__PROJ}
                           POST_BUILD
                           WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${CLDNN_BUILD__PROJ}.exe -ust -hpa )
        endif()
     endif()
endif()

if(UNIX AND WITH_CAFFE)
    add_subdirectory (caffe EXCLUDE_FROM_ALL)
endif()

# ======================================================================================================
