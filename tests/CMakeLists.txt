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

add_subdirectory (gtest)


set (APP_NAME "tests")
# Sources for test files

file (GLOB TEST_CASES_SRC
      test_cases/*.cpp)

# Main source file
set  (MAIN_SRC
      main.cpp
      )

file (GLOB API_HEADERS
       ${CMAKE_SOURCE_DIR}/api/*.h
      )


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("" FILES ${MAIN_SRC})
source_group("test caseses" FILES ${TEST_CASES_SRC})

# Properties->C/C++->General->Additional Include Directories
include_directories (${CMAKE_SOURCE_DIR}
                     ${CMAKE_SOURCE_DIR}/src
                     ${CMAKE_SOURCE_DIR}/tests
                    )

# Create exe file from sources.
add_executable(${APP_NAME}
               ${MAIN_SRC}
               ${TEST_CASES_SRC}
               ${API_HEADERS}
              )

# To use SDE it has to be installed and its root dir needs to be
# added to PATH env variable.Also USE_SDE_EMULATION should be set
# to ON (see scripts for generating build system)
if(USE_SDE_EMULATION)
find_program(SDE_EXE_DIR ${SDE_BINARY_NAME})
message("SDE: ${SDE_EXE_DIR}")
endif()

if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
set(TEST_COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME})
else()
set(TEST_COMMAND ${SDE_EXE_DIR} -bdw -- ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${APP_NAME})
endif()

set_target_properties( ${APP_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set(GFLAGS_BINARY_NAME gflags)

if(RUN_ULTS_OFFLINE)
    add_test(${APP_NAME} ${TEST_COMMAND})
    message (STATUS "add_test ...................... " ${APP_NAME} ${TEST_COMMAND})
else()
    if(NOT UNIX)
        find_program(GFLAGS_EXE_DIR ${GFLAGS_BINARY_NAME})
        message("GFLAGS: ${GFLAGS_EXE_DIR}")

        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${APP_NAME}
                           POST_BUILD
                           WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${APP_NAME}.exe +ust +hpa )
        endif()
    endif()

    if((SDE_EXE_DIR STREQUAL "SDE_EXE_DIR-NOTFOUND") OR (NOT USE_SDE_EMULATION))
        add_custom_command(TARGET ${APP_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                       COMMAND ${APP_NAME} )
    else()
        add_custom_command(TARGET ${APP_NAME}
                       POST_BUILD
                       WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                       COMMAND ${APP_NAME} )
    endif()

    if(NOT UNIX)
        if(NOT(GFLAGS_EXE_DIR STREQUAL "GFLAGS_EXE_DIR-NOTFOUND"))
            add_custom_command(TARGET ${APP_NAME}
                           POST_BUILD
                           WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           COMMAND ${GFLAGS_EXE_DIR} /i ${APP_NAME}.exe -ust -hpa )
        endif()
     endif()
endif()
# Set library dependencies
target_link_libraries(${APP_NAME} gtest neuralia)

if(UNIX AND WITH_CAFFE)
    add_subdirectory (caffe EXCLUDE_FROM_ALL)
endif()

